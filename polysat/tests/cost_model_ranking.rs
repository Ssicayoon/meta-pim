use isl_rs::{Context, Schedule, ScheduleNodeType};
use polysat::optimize::ScheduleCost;
use polysat::schedule_properties::ScheduleProperties;

fn create_schedule(ctx: &Context, _domain_str: &str, _schedule_str: &str) -> Schedule {
    // ISL Schedule::read_from_str expects a YAML representation.
    // We construct a simple YAML schedule for the domain { S[i,j] : 0 <= i,j < 100 }
    // and schedule { S[i,j] -> [i, j] }.
    let yaml = r#"
domain: "{ S[i,j] : 0 <= i,j < 100 }"
child:
  schedule: "[{ S[i,j] -> [i] }, { S[i,j] -> [j] }]"
  child:
    sequence:
    - filter: "{ S[i,j] }"
"#;
    Schedule::read_from_str(ctx, yaml)
}

#[test]
fn test_cost_model_ranking() {
    let ctx = Context::alloc();
    let cost_model = ScheduleCost::default();

    // 1. Sequential Schedule (Baseline)
    // Domain: { S[i, j] : 0 <= i, j < 100 }
    // Schedule: { S[i, j] -> [i, j] }
    // No coincident members.
    let domain_str = "{ S[i, j] : 0 <= i, j < 100 }";
    let sched_str = "{ S[i, j] -> [i, j] }";
    let sched_seq = create_schedule(&ctx, domain_str, sched_str);
    let props_seq = ScheduleProperties::from_isl(&sched_seq);
    let cost_seq = cost_model.compute_cost_from_properties(&props_seq);

    // 2. Parallel Schedule (Flat)
    // Same schedule, but mark outer dimension 'i' as coincident (parallel).
    let sched_par = create_schedule(&ctx, domain_str, sched_str);
    let root = sched_par.get_root();
    // Navigate to the band node. Structure is typically: Domain -> Band
    let band_node = root.child(0);

    // Verify we are at a band node
    assert_eq!(band_node.get_type(), ScheduleNodeType::Band);

    // Mark first dimension (0) as coincident (1 = true)
    let band_node = band_node.band_member_set_coincident(0, 1);

    // We need to reconstruct the schedule from the modified node.
    // In ISL, modifying a node returns a new node that is part of a new tree.
    // We can get the schedule from the node.
    let sched_par = band_node.get_schedule();
    let props_par = ScheduleProperties::from_isl(&sched_par);
    let cost_par = cost_model.compute_cost_from_properties(&props_par);

    // 3. Nested Parallel Schedule (Better)
    // Mark both 'i' and 'j' as coincident.
    let sched_nested = create_schedule(&ctx, domain_str, sched_str);
    let root = sched_nested.get_root();
    let band_node = root.child(0);
    let band_node = band_node.band_member_set_coincident(0, 1);
    let band_node = band_node.band_member_set_coincident(1, 1);
    let sched_nested = band_node.get_schedule();
    let props_nested = ScheduleProperties::from_isl(&sched_nested);
    let cost_nested = cost_model.compute_cost_from_properties(&props_nested);

    println!("Cost Sequential: {}", cost_seq);
    println!("Cost Parallel: {}", cost_par);
    println!("Cost Nested: {}", cost_nested);

    // Verify properties extraction
    println!("Props Seq: {:?}", props_seq);
    println!("Props Par: {:?}", props_par);
    println!("Props Nested: {:?}", props_nested);

    assert_eq!(
        props_seq.parallel_nesting_depth, 0,
        "Sequential should have 0 parallel depth"
    );
    assert_eq!(
        props_par.parallel_nesting_depth, 1,
        "Parallel should have 1 parallel depth"
    );
    assert_eq!(
        props_nested.parallel_nesting_depth, 2,
        "Nested should have 2 parallel depth"
    );

    // Verify ranking: Nested < Parallel < Sequential
    assert!(
        cost_par < cost_seq,
        "Parallel should be cheaper than Sequential"
    );
    assert!(
        cost_nested < cost_par,
        "Nested Parallel should be cheaper than Flat Parallel"
    );
}
