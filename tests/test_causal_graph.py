from mlfactory.experiments.causal_graph.generator import WORLD_IDS, generate_task, regenerate_matches, verify_task


def test_causal_graph_replay_and_verifier():
    task = generate_task(
        17, depth=8, relevant_nodes=11, distractor_nodes=3,
        binary_gate_count=3, negation_count=2, source_update_count=2,
        world_id="greenhouse", render_template_id=1,
    )
    report = verify_task(task)
    assert report["valid"]
    assert report["depth"] == 8
    assert regenerate_matches(task)


def test_all_worlds_have_deterministic_tasks():
    for index, world in enumerate(WORLD_IDS):
        task = generate_task(100 + index, depth=3, binary_gate_count=1, negation_count=1, world_id=world)
        assert verify_task(task)["valid"]
        assert task["world_id"] == world
