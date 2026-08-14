#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import threading
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
    def __init__(self,path:Path,proposals_path:Path|None=None):
        self.path=path
        self.lock=threading.Lock()
        with path.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle)
            self.fields=list(reader.fieldnames or [])
            self.rows=list(reader)
        self.proposals={}
        if proposals_path is not None and proposals_path.is_file():
            with proposals_path.open(newline="",encoding="utf-8") as handle:
                self.proposals={row["video_id"]:row for row in csv.DictReader(handle)}
    @staticmethod
    def video_url(index:int,row:dict)->str:
        digest=row.get("media_sha256","")[:12]
        return f"/video/{index}?v={digest}"
    def _public_row(self,index:int,row:dict):
        return {
            **row,
            **self.proposals.get(row.get("video_id",""),{}),
            "index":index,
            "video_url":self.video_url(index,row),
        }
    def public(self):
        return [self._public_row(i,row) for i,row in enumerate(self.rows)]
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

HTML="""<!doctype html><meta charset="utf-8"><title>FallVision Annotator</title>
<style>body{font:15px sans-serif;background:#111;color:#eee;margin:20px}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}video{width:100%;max-height:70vh;background:#000}input,select,textarea,button{margin:4px;padding:6px}label{display:block}.bad{color:#f77}.ok{color:#7e8}</style>
<h2>FallVision 간단 검토</h2><main><section><select id="pick"></select><div id="meta"></div><video id="v" controls></video><button onclick="step(-1)">-1 frame</button><button onclick="step(1)">+1 frame</button><span id="now"></span></section><section><p><b>영상과 자동 표시가 대략 맞으면 아래 버튼 하나만 누르세요.</b></p><div id="proposal"></div><button id="accept" onclick="acceptProposal()">제안값 승인하고 다음</button><button onclick="excludeAndNext()">영상 오류/판단 불가</button><details><summary>제안값이 틀릴 때만 상세 수정</summary><div id="inputs"></div><label>Status <select id="status"><option>unreviewed</option><option>in_progress</option><option>complete</option><option>needs_adjudication</option><option>excluded</option></select></label><label>Confidence <select id="confidence"><option value=""></option><option>low</option><option>medium</option><option>high</option></select></label><label>Annotator <input id="annotator"></label><label>Notes <textarea id="notes"></textarea></label><button onclick="save(false)">저장</button><button onclick="save(true)">저장하고 다음</button></details><div id="message"></div></section></main>
<script>
const fields=["fall_onset_frame","impact_frame","post_fall_stable_frame","fall_end_frame","onset_earliest_frame","onset_latest_frame"];const proposalMap={fall_onset_frame:"proposed_fall_onset_frame",impact_frame:"proposed_impact_frame",post_fall_stable_frame:"proposed_post_fall_stable_frame",fall_end_frame:"proposed_fall_end_frame",onset_earliest_frame:"proposed_onset_earliest_frame",onset_latest_frame:"proposed_onset_latest_frame"};let items=[],idx=0;const v=document.getElementById("v"),pick=document.getElementById("pick"),meta=document.getElementById("meta"),status=document.getElementById("status"),confidence=document.getElementById("confidence"),annotator=document.getElementById("annotator"),notes=document.getElementById("notes"),inputs=document.getElementById("inputs"),proposal=document.getElementById("proposal"),message=document.getElementById("message"),now=document.getElementById("now");
function frame(){let r=items[idx];return Math.max(0,Math.min(+r.frame_count-1,Math.round(v.currentTime*+r.fps)))}
function step(n){v.currentTime=Math.max(0,v.currentTime+n/(+items[idx].fps))}
function setf(f){document.getElementById(f).value=frame()}
function applyProposal(){let r=items[idx];fields.forEach(f=>{let value=r[proposalMap[f]];if(value!==undefined&&value!=="")document.getElementById(f).value=value});status.value="in_progress";message.className="ok";message.textContent="proposal copied; review video before Save"}
function acceptProposal(){applyProposal();status.value="complete";confidence.value="medium";if(!annotator.value)annotator.value="manual_review";save(true)}
function excludeAndNext(){fields.forEach(f=>document.getElementById(f).value="");status.value="excluded";confidence.value="";notes.value="video error or cannot adjudicate";save(true)}
function load(i){idx=+i;let r=items[idx];v.src=r.video_url;v.load();meta.textContent=r.video_id+" | "+r.fps+" fps | "+r.frame_count+" frames";status.value=r.annotation_status;confidence.value=r.annotation_confidence;annotator.value=r.annotator;notes.value=r.notes;proposal.replaceChildren();if(r.proposed_fall_onset_frame!==undefined){let text=document.createElement("div"),button=document.createElement("button");text.textContent="AUTO REVIEW ONLY — onset "+r.proposed_fall_onset_frame+", impact "+r.proposed_impact_frame+", stable "+r.proposed_post_fall_stable_frame+", end "+r.proposed_fall_end_frame;button.type="button";button.textContent="Apply proposal";button.onclick=applyProposal;proposal.append(text,button)}inputs.replaceChildren();fields.forEach(f=>{let label=document.createElement("label"),input=document.createElement("input"),button=document.createElement("button");label.append(f+" ");input.type="number";input.id=f;input.value=r[f]||"";button.type="button";button.textContent="current";button.onclick=()=>setf(f);label.append(input,button);inputs.append(label)})}
function val(f){let x=document.getElementById(f).value;return x===""?null:+x}
async function save(go){let body={annotation_status:status.value,annotation_confidence:confidence.value,annotator:annotator.value,notes:notes.value};fields.forEach(f=>body[f]=val(f));let res=await fetch("/api/items/"+idx,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});let data=await res.json();message.className=res.ok?"ok":"bad";message.textContent=res.ok?"saved":JSON.stringify(data);if(res.ok){items[idx]=data;if(go){pick.value=Math.min(items.length-1,idx+1);load(pick.value)}}}
v.ontimeupdate=()=>now.textContent="frame "+frame()+" | "+v.currentTime.toFixed(3)+" sec";v.onerror=()=>{message.className="bad";message.textContent="video error: "+(v.error?v.error.code:"unknown")+" / "+v.currentSrc};
fetch("/api/items").then(r=>r.json()).then(data=>{items=data;data.forEach(r=>pick.innerHTML+="<option value="+r.index+">"+r.index+" | "+r.scene_id+" | "+r.recording_id+" | "+r.annotation_status+"</option>");pick.onchange=()=>load(pick.value);load(0)})
</script>"""

def build_app(path:Path,proposals_path:Path|None=None)->FastAPI:
    store=Store(path,proposals_path); app=FastAPI(title="FallVision Annotator")
    @app.get("/",response_class=HTMLResponse)
    def index(): return HTMLResponse(HTML,headers={"Cache-Control":"no-store, max-age=0"})
    @app.get("/api/items")
    def items(): return store.public()
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
    parser.add_argument("--host",default="0.0.0.0");parser.add_argument("--port",type=int,default=8010)
    args=parser.parse_args();uvicorn.run(build_app(args.annotations,args.proposals),host=args.host,port=args.port)
if __name__=="__main__":main()
