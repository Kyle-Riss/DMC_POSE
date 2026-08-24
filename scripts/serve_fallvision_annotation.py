#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

FRAMES=("fall_onset_frame","impact_frame","post_fall_stable_frame","fall_end_frame","onset_earliest_frame","onset_latest_frame")
STATUSES={"unreviewed","in_progress","complete","needs_adjudication","excluded"}
CONFIDENCES={"","low","medium","high"}

class Update(BaseModel):
    fall_onset_frame:int|None=None
    impact_frame:int|None=None
    post_fall_stable_frame:int|None=None
    fall_end_frame:int|None=None
    onset_earliest_frame:int|None=None
    onset_latest_frame:int|None=None
    annotation_status:str="in_progress"
    annotation_confidence:str=""
    annotator:str=""
    notes:str=""

class IdentityUpdate(BaseModel):
    subject_id:str=""
    session_id:str=""
    split:str=""

def validate_annotation(value:Update,frame_count:int)->list[str]:
    data=value.model_dump()
    errors=[]
    for field in FRAMES:
        number=data[field]
        if number is not None and not 0<=number<frame_count:
            errors.append(f"{field} outside 0..{frame_count-1}")
    a,b,c=value.onset_earliest_frame,value.fall_onset_frame,value.onset_latest_frame
    if a is not None and b is not None and a>b: errors.append("earliest must be <= onset")
    if b is not None and c is not None and b>c: errors.append("onset must be <= latest")
    ordered=[value.fall_onset_frame,value.impact_frame,value.post_fall_stable_frame,value.fall_end_frame]
    known=[x for x in ordered if x is not None]
    if known!=sorted(known): errors.append("onset <= impact <= stable <= end required")
    if value.annotation_status not in STATUSES: errors.append("invalid status")
    if value.annotation_confidence not in CONFIDENCES: errors.append("invalid confidence")
    if value.annotation_status=="complete" and any(data[x] is None for x in FRAMES[:4]):
        errors.append("complete requires onset, impact, stable, end")
    return errors

class Store:
    def __init__(self,path:Path,proposals_path:Path|None=None,identity_path:Path|None=None):
        self.path=path
        self.identity_path=identity_path
        self.lock=threading.Lock()
        with path.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle)
            self.fields=list(reader.fieldnames or [])
            self.rows=list(reader)
        self.proposals={}
        if proposals_path is not None and proposals_path.is_file():
            with proposals_path.open(newline="",encoding="utf-8") as handle:
                self.proposals={row["video_id"]:row for row in csv.DictReader(handle)}
        self.identity={"schema_version":"dmc_staged_identity_split_v1","recordings":{}}
        if identity_path is not None and identity_path.is_file():
            self.identity=json.loads(identity_path.read_text(encoding="utf-8"))
    @staticmethod
    def video_url(index:int,row:dict)->str:
        digest=row.get("media_sha256","")[:12]
        return f"/video/{index}?v={digest}"
    def _training_gate(self,row:dict,identity:dict)->tuple[bool,str]:
        blockers=[]
        if row.get("annotation_status")!="complete" or any(
            not str(row.get(field) or "").strip() for field in FRAMES[:4]
        ):
            blockers.append("temporal_annotation_incomplete")
        recording=row.get("recording_id")
        if recording and any(
            candidate.get("annotation_status")!="complete"
            for candidate in self.rows
            if candidate.get("recording_id")==recording
        ):
            blockers.append("multiview_recording_incomplete")
        if not str(identity.get("subject_id") or "").strip():
            blockers.append("subject_identity_unknown")
        if not str(identity.get("session_id") or "").strip():
            blockers.append("session_identity_unknown")
        if str(identity.get("split") or "").strip() not in {"train","val","test"}:
            blockers.append("split_assignment_unknown")
        return not blockers,";".join(blockers)
    def _public_row(self,index:int,row:dict):
        identity=(self.identity.get("recordings") or {}).get(row.get("recording_id"),{})
        eligible,blockers=self._training_gate(row,identity)
        return {
            **row,
            **self.proposals.get(row.get("video_id",""),{}),
            "training_eligible":"true" if eligible else "false",
            "training_blockers":blockers,
            "identity":identity,
            "index":index,
            "video_url":self.video_url(index,row),
        }
    def public(self):
        return [self._public_row(i,row) for i,row in enumerate(self.rows)]
    def recording_indices(self, recording_id:str):
        return [i for i,row in enumerate(self.rows) if row.get("recording_id")==recording_id]
    def progress(self):
        counts=Counter((row.get("annotation_status") or "unreviewed") for row in self.rows)
        recordings=defaultdict(list)
        for row in self.rows:
            recordings[row.get("recording_id") or "unknown"].append(row.get("annotation_status") or "unreviewed")
        complete_recordings=sum(all(status=="complete" for status in statuses) for statuses in recordings.values())
        resolved_recordings=sum(all(status in {"complete","excluded"} for status in statuses) for statuses in recordings.values())
        reviewed_statuses={"complete","needs_adjudication","excluded"}
        reviewed_recordings=sum(all(status in reviewed_statuses for status in statuses) for statuses in recordings.values())
        next_unresolved=next(
            (i for i,row in enumerate(self.rows) if row.get("annotation_status")=="unreviewed"),
            next((i for i,row in enumerate(self.rows) if row.get("annotation_status")=="in_progress"),None),
        )
        return {
            "views_total":len(self.rows),
            "status_counts":dict(sorted(counts.items())),
            "views_complete":counts.get("complete",0),
            "views_reviewed":sum(counts.get(status,0) for status in reviewed_statuses),
            "views_resolved":counts.get("complete",0)+counts.get("excluded",0),
            "recordings_total":len(recordings),
            "recordings_complete":complete_recordings,
            "recordings_reviewed":reviewed_recordings,
            "recordings_resolved":resolved_recordings,
            "recordings_identity_complete":sum(
                bool(value.get("subject_id") and value.get("session_id") and value.get("split") in {"train","val","test"})
                for value in (self.identity.get("recordings") or {}).values()
            ),
            "next_unresolved_index":next_unresolved,
        }
    def update_identity(self,recording_id:str,value:IdentityUpdate):
        mappings=self.identity.setdefault("recordings",{})
        if recording_id not in mappings: raise KeyError(recording_id)
        subject=value.subject_id.strip();session=value.session_id.strip();split=value.split.strip()
        if split and split not in {"train","val","test"}: raise ValueError("split must be train, val, test, or blank")
        if split and (not subject or not session): raise ValueError("split requires subject_id and session_id")
        for other_id,other in mappings.items():
            if other_id==recording_id: continue
            if subject and str(other.get("subject_id") or "").strip()==subject:
                other_split=str(other.get("split") or "").strip()
                if split and other_split and split!=other_split:
                    raise ValueError(f"subject {subject} already belongs to split {other_split}")
        current=mappings[recording_id]
        current["subject_id"]=subject or None
        current["session_id"]=session or None
        current["split"]=split or None
        if self.identity_path is not None:
            with self.lock:
                temp=self.identity_path.with_suffix(self.identity_path.suffix+".tmp")
                temp.write_text(json.dumps(self.identity,ensure_ascii=False,indent=2),encoding="utf-8")
                temp.replace(self.identity_path)
        return current
    def update(self,index:int,value:Update):
        if not 0<=index<len(self.rows): raise IndexError
        row=self.rows[index]
        errors=validate_annotation(value,int(float(row["frame_count"])))
        if errors: raise ValueError(errors)
        data=value.model_dump()
        for field in FRAMES: row[field]="" if data[field] is None else str(data[field])
        for field in ("annotation_status","annotation_confidence","annotator","notes"): row[field]=str(data[field])
        with self.lock:
            temp=self.path.with_suffix(".csv.tmp")
            with temp.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=self.fields)
                writer.writeheader(); writer.writerows(self.rows)
            temp.replace(self.path)
        return self._public_row(index,row)

HTML="""<!doctype html><meta charset="utf-8"><title>DMC staged-fall annotator</title>
<style>body{font:15px sans-serif;background:#111;color:#eee;margin:20px}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}video{width:100%;max-height:70vh;background:#000}input,select,textarea,button{margin:4px;padding:6px}label{display:block}.bad{color:#f77}.ok{color:#7e8}</style>
<h2>DMC 병실 staged fall 검토</h2><div id="progress"></div><button onclick="nextUnresolved()">다음 미검토</button><button onclick="nextView()">같은 사건의 다음 카메라</button><main><section><select id="pick"></select><div id="meta"></div><video id="v" controls></video><button onclick="step(-1)">-1 frame</button><button onclick="step(1)">+1 frame</button><span id="now"></span></section><section><p><b>각 카메라 영상을 직접 확인하세요. 같은 recording이라도 frame 번호를 복사하지 않습니다.</b></p><details open><summary>사건 identity (자동 추정 없음)</summary><label>Subject <input id="subject"></label><label>Session <input id="session"></label><label>Split <select id="split"><option value=""></option><option>train</option><option>val</option><option>test</option></select></label><button onclick="saveIdentity()">Identity 저장</button></details><div id="proposal"></div><button id="accept" onclick="acceptProposal()">제안값 승인하고 다음</button><button onclick="excludeAndNext()">영상 오류/판단 불가</button><details open><summary>시간 경계 입력</summary><div id="inputs"></div><label>Status <select id="status"><option>unreviewed</option><option>in_progress</option><option>complete</option><option>needs_adjudication</option><option>excluded</option></select></label><label>Confidence <select id="confidence"><option value=""></option><option>low</option><option>medium</option><option>high</option></select></label><label>Annotator <input id="annotator"></label><label>Notes <textarea id="notes"></textarea></label><button onclick="save(false)">저장</button><button onclick="save(true)">저장하고 다음 미검토</button></details><div id="message"></div></section></main>
<script>
const fields=["fall_onset_frame","impact_frame","post_fall_stable_frame","fall_end_frame","onset_earliest_frame","onset_latest_frame"];const proposalMap={fall_onset_frame:"proposed_fall_onset_frame",impact_frame:"proposed_impact_frame",post_fall_stable_frame:"proposed_post_fall_stable_frame",fall_end_frame:"proposed_fall_end_frame",onset_earliest_frame:"proposed_onset_earliest_frame",onset_latest_frame:"proposed_onset_latest_frame"};let items=[],idx=0;const v=document.getElementById("v"),pick=document.getElementById("pick"),meta=document.getElementById("meta"),status=document.getElementById("status"),confidence=document.getElementById("confidence"),annotator=document.getElementById("annotator"),notes=document.getElementById("notes"),inputs=document.getElementById("inputs"),proposal=document.getElementById("proposal"),message=document.getElementById("message"),now=document.getElementById("now"),progress=document.getElementById("progress"),accept=document.getElementById("accept"),subject=document.getElementById("subject"),session=document.getElementById("session"),split=document.getElementById("split");
function frame(){let r=items[idx];return Math.max(0,Math.min(+r.frame_count-1,Math.round(v.currentTime*+r.fps)))}
function step(n){v.currentTime=Math.max(0,v.currentTime+n/(+items[idx].fps))}
function setf(f){document.getElementById(f).value=frame()}
function refreshProgress(){fetch("/api/progress").then(r=>r.json()).then(x=>{progress.textContent=`영상 검토 ${x.views_reviewed}/${x.views_total} (완료 ${x.views_complete}, 보류 ${x.status_counts.needs_adjudication||0}) · 사건 검토 ${x.recordings_reviewed}/${x.recordings_total} (완료 ${x.recordings_complete}) · identity ${x.recordings_identity_complete}/${x.recordings_total}`})}
function nextUnresolved(){let next=items.findIndex((r,i)=>i>idx&&r.annotation_status==='unreviewed');if(next<0)next=items.findIndex(r=>r.annotation_status==='unreviewed');if(next<0)next=items.findIndex(r=>r.annotation_status==='in_progress');if(next>=0){pick.value=next;load(next)}}
function nextView(){let group=items.filter(r=>r.recording_id===items[idx].recording_id).map(r=>r.index),at=group.indexOf(idx),next=group[(at+1)%group.length];pick.value=next;load(next)}
function applyProposal(){let r=items[idx];fields.forEach(f=>{let value=r[proposalMap[f]];if(value!==undefined&&value!=="")document.getElementById(f).value=value});status.value="in_progress";message.className="ok";message.textContent="proposal copied; review video before Save"}
function acceptProposal(){applyProposal();status.value="complete";confidence.value="medium";if(!annotator.value)annotator.value="manual_review";save(true)}
function excludeAndNext(){fields.forEach(f=>document.getElementById(f).value="");status.value="excluded";confidence.value="";notes.value="video error or cannot adjudicate";save(true)}
function load(i){idx=+i;let r=items[idx],identity=r.identity||{};v.src=r.video_url;v.load();meta.textContent=r.recording_id+" | "+r.camera_id+" | "+r.fps+" fps | "+r.frame_count+" frames";subject.value=identity.subject_id||"";session.value=identity.session_id||"";split.value=identity.split||"";status.value=r.annotation_status;confidence.value=r.annotation_confidence;annotator.value=r.annotator;notes.value=r.notes;proposal.replaceChildren();accept.disabled=true;if(r.proposed_fall_onset_frame!==undefined){let adjudication=r.multiview_status==="needs_adjudication";accept.disabled=adjudication;let text=document.createElement("div"),button=document.createElement("button");text.className=adjudication?"bad":"ok";text.textContent="AUTO REVIEW ONLY — "+(r.multiview_status||"single-view")+" | onset "+r.proposed_fall_onset_frame+", impact "+r.proposed_impact_frame+", stable "+r.proposed_post_fall_stable_frame+", end "+r.proposed_fall_end_frame+(r.onset_spread_sec!==undefined?" | spreads "+r.onset_spread_sec+"/"+r.impact_spread_sec+"/"+r.stable_spread_sec+" sec":"");button.type="button";button.textContent=adjudication?"Apply for manual correction":"Apply proposal";button.onclick=applyProposal;proposal.append(text,button)}inputs.replaceChildren();fields.forEach(f=>{let label=document.createElement("label"),input=document.createElement("input"),button=document.createElement("button");label.append(f+" ");input.type="number";input.id=f;input.value=r[f]||"";button.type="button";button.textContent="current";button.onclick=()=>setf(f);label.append(input,button);inputs.append(label)})}
function val(f){let x=document.getElementById(f).value;return x===""?null:+x}
async function save(go){let body={annotation_status:status.value,annotation_confidence:confidence.value,annotator:annotator.value,notes:notes.value};fields.forEach(f=>body[f]=val(f));let res=await fetch("/api/items/"+idx,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});let data=await res.json();message.className=res.ok?"ok":"bad";message.textContent=res.ok?"saved":JSON.stringify(data);if(res.ok){items[idx]=data;refreshProgress();if(go)nextUnresolved()}}
async function saveIdentity(){let recording=items[idx].recording_id,body={subject_id:subject.value,session_id:session.value,split:split.value};let res=await fetch("/api/identity/"+encodeURIComponent(recording),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}),data=await res.json();message.className=res.ok?"ok":"bad";message.textContent=res.ok?"identity saved":JSON.stringify(data);if(res.ok){items.filter(r=>r.recording_id===recording).forEach(r=>r.identity=data);refreshProgress()}}
v.ontimeupdate=()=>now.textContent="frame "+frame()+" | "+v.currentTime.toFixed(3)+" sec";v.onerror=()=>{message.className="bad";message.textContent="video error: "+(v.error?v.error.code:"unknown")+" / "+v.currentSrc};
fetch("/api/items").then(r=>r.json()).then(data=>{items=data;data.forEach(r=>pick.innerHTML+="<option value="+r.index+">"+r.index+" | "+r.recording_id+" | "+r.camera_id+" | "+r.annotation_status+"</option>");pick.onchange=()=>load(pick.value);refreshProgress();load(0)})
</script>"""

def build_app(path:Path,proposals_path:Path|None=None,identity_path:Path|None=None)->FastAPI:
    store=Store(path,proposals_path,identity_path); app=FastAPI(title="DMC staged-fall annotator")
    @app.get("/",response_class=HTMLResponse)
    def index(): return HTMLResponse(HTML,headers={"Cache-Control":"no-store, max-age=0"})
    @app.get("/api/items")
    def items(): return store.public()
    @app.get("/api/progress")
    def progress(): return store.progress()
    @app.get("/api/recordings/{recording_id}")
    def recording(recording_id:str): return [store.public()[i] for i in store.recording_indices(recording_id)]
    @app.post("/api/identity/{recording_id}")
    def identity(recording_id:str,value:IdentityUpdate):
        try: return store.update_identity(recording_id,value)
        except KeyError: raise HTTPException(404)
        except ValueError as error: raise HTTPException(422,detail=str(error))
    @app.get("/video/{index}")
    def video(index:int):
        if not 0<=index<len(store.rows): raise HTTPException(404)
        path=Path(store.rows[index]["local_video_path"])
        if not path.is_file(): raise HTTPException(404,"media missing")
        return FileResponse(path,media_type="video/mp4",headers={"Cache-Control":"no-store, max-age=0"})
    @app.post("/api/items/{index}")
    def update(index:int,value:Update):
        try: return store.update(index,value)
        except IndexError: raise HTTPException(404)
        except ValueError as error: raise HTTPException(422,detail=error.args[0])
    return app

def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--annotations",type=Path,default=project/"external_datasets/annotations/fallvision_pilot_v1.csv")
    parser.add_argument("--proposals",type=Path,default=project/"external_datasets/annotations/fallvision_pilot_v1_proposals.csv")
    parser.add_argument("--identity",type=Path,default=project/"external_datasets/annotations/usb_sim_falldown_identity_v1.json")
    parser.add_argument("--host",default="0.0.0.0");parser.add_argument("--port",type=int,default=8010)
    args=parser.parse_args();uvicorn.run(build_app(args.annotations,args.proposals,args.identity),host=args.host,port=args.port)
if __name__=="__main__":main()
