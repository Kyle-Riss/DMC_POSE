from scripts.build_multiview_consensus_proposals import build_consensus


def rows(onsets):
    annotations=[];proposals=[]
    for index,onset in enumerate(onsets,1):
        video=f"v{index}";camera=f"c{index}"
        annotations.append({"video_id":video,"recording_id":"r1","camera_id":camera,"fps":"20","frame_count":"240"})
        proposals.append({"video_id":video,"recording_id":"r1","scene_id":camera,"proposed_fall_onset_frame":str(onset),"proposed_impact_frame":"100","proposed_post_fall_stable_frame":"120"})
    return annotations,proposals


def test_consensus_uses_median_and_preserves_three_views():
    output,report=build_consensus(*rows([40,42,44]))
    assert len(output)==3
    assert {row["proposed_fall_onset_frame"] for row in output}=={42}
    assert report["status_counts"]=={"multiview_consistent":1}


def test_large_camera_disagreement_requires_adjudication():
    output,report=build_consensus(*rows([20,40,80]))
    assert report["status_counts"]=={"needs_adjudication":1}
    assert all(row["multiview_status"]=="needs_adjudication" for row in output)
