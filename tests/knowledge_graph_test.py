from npcpy.memory.knowledge_graph import *
import os
import datetime
import traceback
import random
from typing import Dict, List, Set, Tuple, Optional, Any
from npcpy.npc_compiler import NPC

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_subsection(title):
    print("\n" + "-" * 60)
    print(f" {title} ".center(60, "-"))
    print("-" * 60 + "\n")


def print_breathe_output(output):
    print_subsection("Breathe Output")
    print("Facts:", output.get("facts", []))
    print("Mistakes:", output.get("mistakes", []))
    print("Lessons Learned:", output.get("lessons_learned", []))



def print_hierarchy(hierarchy_data):
    print_subsection("Hierarchy Structure")
    dag = hierarchy_data.get("dag", {})
    
    # Get all levels
    level_groups = {}
    for group, data in dag.items():
        level = data.get("level", 0)
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(group)
    
    print(f"Total groups in DAG: {len(dag)}")
    
    # Print each level and its relationships
    for level in sorted(level_groups.keys(), reverse=True):
        print(f"\nLevel {level}:")
        for group in level_groups[level]:
            children = dag[group]["children"]
            parents = dag[group]["parents"]
            print(f"  {group}")
            if parents:
                print(f"    Parents: {', '.join(parents)}")
            if children:
                print(f"    Children: {', '.join(children)}")
            
    # Print group relationships
    print("\nSemantic Evolution:")
    for level in sorted(level_groups.keys()):
        groups = level_groups[level]
        print(f"\nLevel {level} Abstractions:")
        for group in groups:
            children = dag[group]["children"]
            if children:
                print(f"  {group} abstracts:")
                for child in children:
                    print(f"    - {child}")

def print_assignments(assignments):
    print_subsection("Fact Assignments")
    sorted_facts = sorted(assignments.keys())
    for fact in sorted_facts:
        assignment = assignments[fact]
        print(f"Fact: {fact[:80]}...")
        print(f"  Direct Leaf Groups: {assignment.get('direct_groups', [])}")
        print(f"  All Groups (incl. ancestors): {assignment.get('all_groups', [])}")
        print(f"  Hierarchy Paths: {assignment.get('hierarchy_paths', [])}")


def get_db_counts(conn):
     fact_count_result = conn.execute("MATCH (f:Fact) RETURN count(f) as count").get_as_df()
     fact_count = fact_count_result.iloc[0]["count"]

     group_count_result = conn.execute("MATCH (g:Groups) RETURN count(g) as count").get_as_df()
     group_count = group_count_result.iloc[0]["count"]

     rel_count_result = conn.execute("MATCH ()-[r:Contains]->() RETURN count(r) as count").get_as_df()
     rel_count = rel_count_result.iloc[0]["count"]

     return {"facts": fact_count, "groups": group_count, "relationships": rel_count}
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(os.path.expanduser("~"), f"kg_test_db_{timestamp}")

    model = "gpt-4o-mini"
    provider = "openai"
    npc = NPC(name='emir',
              primary_directive='read and understand',
              model=model,
              provider=provider)


    print(f"🚀 Starting Knowledge Graph Integration Test")
    print(f"📁 Database path: {db_path}")
    print(f"🤖 Model: {model} / Provider: {provider}")

    if os.path.exists(db_path):
         print(f"Removing existing DB directory at {db_path}")
         try:
             import shutil
             shutil.rmtree(db_path)
             print("Directory removed.")
         except Exception as e:
             print(f"Failed to remove directory: {e}")


    conn = init_db(db_path, drop=True)

    current_kg_state: Dict[str, Any] = {
        "facts": [],
        "leaf_groups": [],
        "hierarchy": {},
        "assignments": {}
    }

    all_messages: List[Dict[str, str]] = []

    print_section("SCENARIO 1: INITIAL CONVERSATION - TDE CIRCULARIZATION DYNAMICS")

    messages_1_snippet = [
        {"role": "user", "content": "ok im trying to get my head around tidal disruption events when a star gets too close to a supermassive black hole and gets ripped apart into a gas stream but how does that stream form a disk like it starts on this really stretched out path does gravity just pull it into a circle or is there other stuff going on like hydrodynamics or effects really close to the black hole or what drives the gas coming back in to form a disk"},
        {"role": "assistant", "content": "The circularization process in tidal disruption events is a key area of research. While gravity is responsible for the initial stretching of the star into a stream, the formation of an accretion disk is primarily driven by hydrodynamical effects, specifically self-intersection shocks within the debris stream. As the leading portion of the stream orbits and returns towards the black hole, it collides with the trailing portion still infalling. This collision dissipates orbital energy through shocks, allowing the bound material to settle into more circular orbits and form a nascent accretion disk. Relativistic effects become more significant closer to the black hole but are not the dominant mechanism for this initial circularization."},
        {"role": "user", "content": "so the stream hits itself and that shock heats up the gas and makes it lose energy which helps it circularize okay makes sense. and that shock region is pretty far out from the black hole like tens or hundreds of gravitational radii not super close where weird general relativity stuff is strongest right? and that collision efficiency how much energy it loses that determines how much of the star's gas actually gets captured by the black hole versus escaping or staying on big looping orbits"},
        {"role": "assistant", "content": "That is correct. The self-intersection radius is typically located at tens to hundreds of gravitational radii, well outside the strongly relativistic regime near the event horizon. The energy dissipation in these self-intersection shocks is indeed a critical factor influencing the capture fraction – the proportion of stellar mass that is sufficiently bound to eventually accrete onto the black hole, as opposed to the unbound material which is ejected or remains on wide orbits. This process is a significant determinant of the mass supply available for accretion disk formation."},
    ]

    all_messages.extend(messages_1_snippet)

    print("Running breathe() on cumulative conversation...")
    breathe_output_1 = breathe(all_messages, model, provider, npc)
    print_breathe_output(breathe_output_1["output"])

    extracted_facts_1 = breathe_output_1["output"].get("facts", [])


    print("\nRunning semantic_evolution() for the first time to build hierarchy and get leaf groups...")
    evolution_result_1 = semantic_evolution(
        extracted_facts_1,
        current_kg_state.get("leaf_groups", []),
        model,
        provider,
        npc
    )

    current_kg_state["leaf_groups"] = evolution_result_1.get("leaf_groups", [])
    current_kg_state["hierarchy"] = evolution_result_1.get("hierarchy", {})
    current_kg_state["facts"].extend(extracted_facts_1)

    # CREATE GROUPS AND STORE FACTS IN DB
    for group in evolution_result_1.get("hierarchy", {}).get("dag", {}).keys():
        create_group(conn, group, "")

    for fact in extracted_facts_1:
        assignment = assign_fact_to_dag(fact, current_kg_state["hierarchy"], model, provider, npc)
        store_fact_and_group(conn, fact, assignment["all_groups"], "")
        current_kg_state["assignments"][fact] = assignment


    print("\n--- Results after Scenario 1 ---")
    print_hierarchy(current_kg_state.get("hierarchy", {}))
    print_assignments(current_kg_state.get("assignments", {}))
    print("DB Counts after Scenario 1:", get_db_counts(conn))


    # ========================================================================
    # TEST SCENARIO 2: CONTINUING CONVERSATION - TDE FALLBACK AND OBSERVATIONS
    # ========================================================================
    print_section("SCENARIO 2: CONTINUING CONVERSATION - TDE FALLBACK AND OBSERVATIONS")

    messages_2_snippet = [
        {"role": "user", "content": "okay so self-intersection shocks do the initial circularization far out got it. then that bound gas falls back onto the black hole right and thats what makes the flares we see the lightcurves. the simplest theory predicts the fallback rate follows like a time to the minus five-thirds power law t^-5/3 but real TDE lightcurves dont always follow that perfect shape they are sometimes steeper sometimes shallower so what else could be causing those deviations from the simple model is it just about the type of star or the black hole spin or maybe something else entirely like maybe outflows from the accretion disk are blowing stuff away or obscuring the light"},
        {"role": "assistant", "content": "That is correct. The $t^{-5/3}$ power law is a theoretical prediction derived from simplified models assuming prompt and complete circularization of the debris stream. Observed lightcurves frequently deviate from this idealized decay, which is a significant area of investigation. Several factors are considered potential causes for these deviations. The internal structure and composition of the disrupted star, such as whether it was a main-sequence star, a red giant, or a white dwarf, influences the debris's specific energy distribution and can affect the fallback rate. The spin and mass of the black hole can also impact the spacetime geometry and relativistic effects, potentially altering the stream dynamics closer in. Additionally, physical processes occurring during the accretion phase itself, like the launching of powerful winds or outflows, can obscure the emitted light or modify the mass accretion rate, thus affecting the observed lightcurve shape."},
        {"role": "user", "content": "okay so stellar structure black hole spin outflows and maybe obscuration or viewing angle can cause those lightcurve deviations from the t^-5/3 law. that means understanding tdes from the lightcurve alone is tricky right you probably need other data like spectra or observations in different wavelengths like radio or x-ray or uv to really figure out whats going on like some events even have neutrino detections what does that extra data give you compared to just the lightcurve"},
        {"role": "assistant", "content": "You are correct; relying solely on lightcurves for classifying TDEs or inferring black hole properties is often insufficient due to inherent degeneracies. Distinct combinations of stellar and black hole parameters can produce remarkably similar lightcurve shapes. Spectroscopic observations are crucial for providing definitive evidence of a TDE by revealing broad emission lines from the stellar debris, which also allow for estimates of the gas composition, temperature, and velocity. Multi-wavelength data, encompassing radio, optical, UV, and X-ray observations, are essential as they probe different physical regions and emission mechanisms within the accretion flow and potential outflowing material, offering a more comprehensive view that helps break these degeneracies and constrain theoretical models more effectively than a lightcurve alone. Rare detections of high-energy phenomena like neutrinos or gamma rays from some events suggest the presence of powerful relativistic jets, adding another dimension to the complex picture of TDE phenomenology."},
    ]



    print("Running breathe() on cumulative conversation...")
    breathe_output_2 = breathe(messages_2_snippet, model, provider, npc)
    print_breathe_output(breathe_output_2["output"])

    extracted_facts_2 = breathe_output_2["output"].get("facts", [])


    print("\nRunning semantic_evolution() with new facts and existing groups from Scenario 1...")
    evolution_result_2 = semantic_evolution(
        extracted_facts_2,
        current_kg_state.get("leaf_groups", []),
        model,
        provider,
        npc
    )

    current_kg_state["leaf_groups"] = evolution_result_2.get("leaf_groups", [])
    current_kg_state["hierarchy"] = evolution_result_2.get("hierarchy", {})
    current_kg_state["facts"].extend([f for f in extracted_facts_2 if f not in current_kg_state["facts"]])

    # CREATE NEW GROUPS AND STORE NEW FACTS IN DB
    for group in evolution_result_2.get("hierarchy", {}).get("dag", {}).keys():
        create_group(conn, group, "")

    for fact in extracted_facts_2:
        assignment = assign_fact_to_dag(fact, current_kg_state["hierarchy"], model, provider, npc)
        store_fact_and_group(conn, fact, assignment["all_groups"], "")
        current_kg_state["assignments"][fact] = assignment


    print("\n--- Results after Scenario 2 ---")
    print_hierarchy(current_kg_state.get("hierarchy", {}))
    print_assignments(current_kg_state.get("assignments", {}))
    print("DB Counts after Scenario 2:", get_db_counts(conn))


    # ========================================================================
    # Cleanup
    # ========================================================================
    print_section("TEST COMPLETE")
    if conn:
        conn.close()
        print(f"Database connection closed. Database file at: {db_path}")


