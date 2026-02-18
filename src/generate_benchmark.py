"""
ConsistencyBench: A benchmark for evaluating cross-query logical consistency in LLMs.

Generates ~500 logically entailed question sets across 6 categories:
1. Contrapositive - Tests if models handle P→Q ↔ ¬Q→¬P
2. Transitivity - Tests if models handle A→B, B→C ⊢ A→C
3. Syllogistic - Tests if models handle All A are B, X is A ⊢ X is B
4. Negation - Tests if models give opposite answers to P vs ¬P
5. Modus Tollens - Tests if models handle P→Q, ¬Q ⊢ ¬P
6. Commonsense Entailment - Tests everyday logical entailments
"""

import json
import random
import os
from itertools import product

random.seed(42)

# ============================================================
# ENTITY AND PREDICATE POOLS
# ============================================================

PEOPLE = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Maria", "Nathan", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier"
]

ANIMALS = [
    "dogs", "cats", "birds", "fish", "horses", "rabbits", "dolphins",
    "eagles", "lions", "bears", "wolves", "tigers", "elephants", "whales",
    "penguins", "owls", "snakes", "turtles", "foxes", "deer"
]

PROPERTIES = [
    ("warm-blooded", "cold-blooded"),
    ("herbivore", "carnivore"),
    ("nocturnal", "diurnal"),
    ("domesticated", "wild"),
    ("aquatic", "terrestrial"),
    ("migratory", "non-migratory"),
    ("social", "solitary"),
    ("endangered", "thriving"),
]

ACTIVITIES = [
    ("goes to the gym", "stays home"),
    ("takes the bus", "drives a car"),
    ("eats breakfast", "skips breakfast"),
    ("reads a book", "watches television"),
    ("goes for a walk", "stays indoors"),
    ("cooks dinner", "orders takeout"),
    ("studies mathematics", "plays video games"),
    ("wakes up early", "sleeps in"),
    ("drinks coffee", "drinks tea"),
    ("goes swimming", "goes running"),
]

CONDITIONS = [
    ("it is raining", "it is sunny"),
    ("the temperature is above 30°C", "the temperature is below 10°C"),
    ("it is a weekday", "it is a weekend"),
    ("the store is open", "the store is closed"),
    ("the road is clear", "the road is blocked"),
    ("the library is open", "the library is closed"),
    ("school is in session", "school is on break"),
    ("the flight is on time", "the flight is delayed"),
]

CATEGORIES_ABSTRACT = [
    "mammals", "reptiles", "vehicles", "fruits", "planets",
    "instruments", "metals", "languages", "countries", "rivers"
]

SUPERCATEGORIES = {
    "mammals": "animals", "reptiles": "animals", "vehicles": "machines",
    "fruits": "foods", "planets": "celestial bodies", "instruments": "tools",
    "metals": "elements", "languages": "communication systems",
    "countries": "political entities", "rivers": "water bodies"
}

INSTANCES = {
    "mammals": ["a dog", "a cat", "a whale", "a bat", "an elephant"],
    "reptiles": ["a lizard", "a snake", "a turtle", "a crocodile", "an iguana"],
    "vehicles": ["a car", "a bus", "a truck", "a motorcycle", "a bicycle"],
    "fruits": ["an apple", "a banana", "an orange", "a grape", "a mango"],
    "planets": ["Mars", "Venus", "Jupiter", "Saturn", "Mercury"],
    "instruments": ["a piano", "a guitar", "a violin", "a drum", "a flute"],
    "metals": ["gold", "silver", "iron", "copper", "aluminum"],
    "languages": ["English", "Spanish", "Mandarin", "French", "Arabic"],
    "countries": ["France", "Japan", "Brazil", "Canada", "Australia"],
    "rivers": ["the Nile", "the Amazon", "the Yangtze", "the Mississippi", "the Danube"],
}

ABSTRACT_PROPERTIES = {
    "mammals": ["warm-blooded", "have fur or hair", "nurse their young"],
    "reptiles": ["cold-blooded", "have scales", "lay eggs"],
    "vehicles": ["have wheels", "require fuel or energy", "can transport people"],
    "fruits": ["contain seeds", "grow on plants", "are edible"],
    "planets": ["orbit a star", "have gravity", "are spherical"],
    "instruments": ["produce sound", "require skill to play", "are used in music"],
    "metals": ["conduct electricity", "are malleable", "have luster"],
    "languages": ["have grammar rules", "are used for communication", "have vocabulary"],
    "countries": ["have borders", "have a government", "have a population"],
    "rivers": ["flow to the sea", "contain freshwater", "have a source"],
}


def generate_id():
    """Generate a unique ID for each question set."""
    return f"CB-{random.randint(10000, 99999)}"


# ============================================================
# CATEGORY 1: CONTRAPOSITIVE
# Tests: P → Q entails ¬Q → ¬P
# ============================================================

def generate_contrapositive_sets(n=85):
    """Generate contrapositive question sets."""
    sets = []
    used = set()
    
    for _ in range(n * 3):  # oversample then trim
        if len(sets) >= n:
            break
            
        person = random.choice(PEOPLE)
        cond_pair = random.choice(CONDITIONS)
        act_pair = random.choice(ACTIVITIES)
        
        key = (person, cond_pair[0], act_pair[0])
        if key in used:
            continue
        used.add(key)
        
        P = cond_pair[0]
        Q = act_pair[0]
        not_P = cond_pair[1]
        not_Q = act_pair[1]
        
        premise = f"If {P}, then {person} {Q}."
        
        questions = [
            {
                "question": f"Given the rule: \"{premise}\" If {P}, does {person} {Q}?",
                "expected_answer": "Yes",
                "reasoning": "Direct application of modus ponens: P is true, P→Q, so Q is true."
            },
            {
                "question": f"Given the rule: \"{premise}\" If {person} {not_Q} (i.e., does not {Q.rstrip('.')}), can we conclude that {not_P}?",
                "expected_answer": "Yes",
                "reasoning": "Contrapositive: P→Q is equivalent to ¬Q→¬P."
            },
            {
                "question": f"Given the rule: \"{premise}\" If {not_P}, can we conclude whether {person} {Q} or not?",
                "expected_answer": "No, we cannot conclude either way",
                "reasoning": "Denying the antecedent is a fallacy: ¬P does not tell us about Q."
            },
            {
                "question": f"Given the rule: \"{premise}\" If {person} {Q}, can we conclude that {P}?",
                "expected_answer": "No, we cannot conclude that",
                "reasoning": "Affirming the consequent is a fallacy: Q being true doesn't mean P caused it."
            }
        ]
        
        sets.append({
            "id": generate_id(),
            "category": "contrapositive",
            "premise": premise,
            "questions": questions,
            "num_questions": len(questions),
            "logical_form": f"P→Q where P=\"{P}\", Q=\"{person} {Q}\""
        })
    
    return sets[:n]


# ============================================================
# CATEGORY 2: TRANSITIVITY
# Tests: A→B, B→C entails A→C
# ============================================================

def generate_transitivity_sets(n=85):
    """Generate transitivity question sets."""
    sets = []
    used = set()
    
    # Create chains of conditions/activities
    for _ in range(n * 3):
        if len(sets) >= n:
            break
        
        person = random.choice(PEOPLE)
        acts = random.sample(ACTIVITIES, 3)
        cond = random.choice(CONDITIONS)
        
        A = cond[0]
        B = acts[0][0]
        C = acts[1][0]
        not_C = acts[1][1]
        
        key = (person, A, B, C)
        if key in used:
            continue
        used.add(key)
        
        premise1 = f"If {A}, then {person} {B}."
        premise2 = f"If {person} {B}, then {person} {C}."
        
        questions = [
            {
                "question": f"Given: (1) \"{premise1}\" (2) \"{premise2}\" If {A}, does {person} {B}?",
                "expected_answer": "Yes",
                "reasoning": "Direct application of rule 1."
            },
            {
                "question": f"Given: (1) \"{premise1}\" (2) \"{premise2}\" If {person} {B}, does {person} {C}?",
                "expected_answer": "Yes",
                "reasoning": "Direct application of rule 2."
            },
            {
                "question": f"Given: (1) \"{premise1}\" (2) \"{premise2}\" If {A}, does {person} {C}?",
                "expected_answer": "Yes",
                "reasoning": "Transitivity: A→B and B→C, therefore A→C."
            },
            {
                "question": f"Given: (1) \"{premise1}\" (2) \"{premise2}\" If {person} {not_C}, can we conclude that it is not the case that {A}?",
                "expected_answer": "Yes",
                "reasoning": "Contrapositive of transitive chain: ¬C→¬B→¬A."
            }
        ]
        
        sets.append({
            "id": generate_id(),
            "category": "transitivity",
            "premise": f"{premise1} {premise2}",
            "questions": questions,
            "num_questions": len(questions),
            "logical_form": f"A→B, B→C where A=\"{A}\", B=\"{person} {B}\", C=\"{person} {C}\""
        })
    
    return sets[:n]


# ============================================================
# CATEGORY 3: SYLLOGISTIC
# Tests: All A are B, X is A ⊢ X is B (and variants)
# ============================================================

def generate_syllogistic_sets(n=85):
    """Generate syllogistic reasoning question sets."""
    sets = []
    used = set()
    
    for _ in range(n * 3):
        if len(sets) >= n:
            break
        
        cat = random.choice(CATEGORIES_ABSTRACT)
        supercat = SUPERCATEGORIES[cat]
        instance = random.choice(INSTANCES[cat])
        prop = random.choice(ABSTRACT_PROPERTIES[cat])
        
        key = (cat, instance, prop)
        if key in used:
            continue
        used.add(key)
        
        premise1 = f"All {cat} are {supercat}."
        premise2 = f"All {cat} {prop}."
        premise3 = f"{instance.capitalize()} is a member of {cat}."
        
        questions = [
            {
                "question": f"Given: (1) \"{premise1}\" (2) \"{premise3}\" Is {instance} a member of {supercat}?",
                "expected_answer": "Yes",
                "reasoning": f"Barbara syllogism: All {cat} are {supercat}, {instance} is a {cat.rstrip('s')}, therefore {instance} is a {supercat.rstrip('s')}."
            },
            {
                "question": f"Given: (1) \"{premise2}\" (2) \"{premise3}\" Does {instance} {prop}?",
                "expected_answer": "Yes",
                "reasoning": f"Barbara syllogism: All {cat} {prop}, {instance} is a {cat.rstrip('s')}, therefore {instance} {prop}."
            },
            {
                "question": f"Given: (1) \"{premise1}\" (2) \"{premise2}\" If something is a member of {supercat}, does it necessarily {prop}?",
                "expected_answer": "No, not necessarily",
                "reasoning": f"Illicit major: All {cat} are {supercat} and all {cat} {prop}, but not all {supercat} need to {prop}."
            },
            {
                "question": f"Given: (1) \"{premise1}\" If something is a member of {supercat}, is it necessarily a member of {cat}?",
                "expected_answer": "No, not necessarily",
                "reasoning": f"Converse error: All {cat} are {supercat} does not mean all {supercat} are {cat}."
            }
        ]
        
        sets.append({
            "id": generate_id(),
            "category": "syllogistic",
            "premise": f"{premise1} {premise2} {premise3}",
            "questions": questions,
            "num_questions": len(questions),
            "logical_form": f"∀x: {cat}(x)→{supercat}(x), ∀x: {cat}(x)→{prop}(x), {cat}({instance})"
        })
    
    return sets[:n]


# ============================================================
# CATEGORY 4: NEGATION CONSISTENCY
# Tests: answer to P and answer to ¬P should be opposite
# ============================================================

def generate_negation_sets(n=85):
    """Generate negation consistency question sets."""
    sets = []
    used = set()
    
    # Factual-style negation pairs
    factual_pairs = [
        ("Paris is the capital of France", True),
        ("Water boils at 100°C at standard atmospheric pressure", True),
        ("The Earth orbits around the Moon", False),
        ("Humans have 46 chromosomes", True),
        ("The speed of light is approximately 300,000 km/s", True),
        ("Oxygen is the most abundant element in the universe", False),
        ("DNA stands for deoxyribonucleic acid", True),
        ("The Great Wall of China is visible from space with the naked eye", False),
        ("Pi is a rational number", False),
        ("Sound travels faster in water than in air", True),
        ("Venus is the closest planet to the Sun", False),
        ("The chemical formula for table salt is NaCl", True),
        ("Electrons carry a positive charge", False),
        ("The Amazon River is the longest river in the world", False),
        ("Antibiotics are effective against viruses", False),
        ("Gold is a chemical element with symbol Au", True),
        ("Photosynthesis produces carbon dioxide", False),
        ("Mount Everest is the tallest mountain on Earth", True),
        ("The Atlantic Ocean is the largest ocean", False),
        ("Gravity pulls objects toward the center of the Earth", True),
    ]
    
    for _ in range(n * 3):
        if len(sets) >= n:
            break
        
        person = random.choice(PEOPLE)
        stmt, truth_value = random.choice(factual_pairs)
        
        key = (person, stmt)
        if key in used:
            continue
        used.add(key)
        
        pos_answer = "Yes" if truth_value else "No"
        neg_answer = "No" if truth_value else "Yes"
        
        questions = [
            {
                "question": f"Is the following statement true? \"{stmt}\"",
                "expected_answer": pos_answer,
                "reasoning": f"The statement is {'true' if truth_value else 'false'} based on factual knowledge."
            },
            {
                "question": f"Is the following statement false? \"{stmt}\"",
                "expected_answer": neg_answer,
                "reasoning": f"Since the statement is {'true' if truth_value else 'false'}, asking if it's false should yield {'No' if truth_value else 'Yes'}."
            },
            {
                "question": f"Is the negation of the following statement true? \"It is NOT the case that {stmt.lower()}.\"",
                "expected_answer": neg_answer,
                "reasoning": f"The negation of a {'true' if truth_value else 'false'} statement is {'false' if truth_value else 'true'}."
            }
        ]
        
        # Add a conditional that depends on the truth value
        cond_act = random.choice(ACTIVITIES)
        questions.append({
            "question": f"Suppose that if \"{stmt.lower()}\" is true, then {person} {cond_act[0]}. Given your assessment of the statement, does {person} {cond_act[0]}?",
            "expected_answer": "Yes" if truth_value else "Cannot be determined from the given information",
            "reasoning": f"If the statement is {'true' if truth_value else 'false'}, and the rule triggers on truth, then {'the consequent follows' if truth_value else 'we cannot conclude the consequent (antecedent is false)'}."
        })
        
        sets.append({
            "id": generate_id(),
            "category": "negation",
            "premise": stmt,
            "questions": questions,
            "num_questions": len(questions),
            "logical_form": f"P = \"{stmt}\", truth_value={truth_value}"
        })
    
    return sets[:n]


# ============================================================
# CATEGORY 5: MODUS TOLLENS
# Tests: P→Q, ¬Q ⊢ ¬P
# ============================================================

def generate_modus_tollens_sets(n=85):
    """Generate modus tollens question sets."""
    sets = []
    used = set()
    
    for _ in range(n * 3):
        if len(sets) >= n:
            break
        
        person = random.choice(PEOPLE)
        cond = random.choice(CONDITIONS)
        act = random.choice(ACTIVITIES)
        
        P = cond[0]
        not_P = cond[1]
        Q = f"{person} {act[0]}"
        not_Q = f"{person} {act[1]}"
        
        key = (person, P, Q)
        if key in used:
            continue
        used.add(key)
        
        premise = f"If {P}, then {Q}."
        
        questions = [
            {
                "question": f"Given: \"{premise}\" We observe that {not_Q}. Can we conclude that {not_P}?",
                "expected_answer": "Yes",
                "reasoning": "Modus tollens: P→Q, ¬Q, therefore ¬P."
            },
            {
                "question": f"Given: \"{premise}\" We observe that {Q}. Can we conclude that {P}?",
                "expected_answer": "No, we cannot conclude that",
                "reasoning": "Affirming the consequent fallacy: Q being true doesn't prove P."
            },
            {
                "question": f"Given: \"{premise}\" We observe that {not_P}. Can we conclude that {not_Q}?",
                "expected_answer": "No, we cannot conclude that",
                "reasoning": "Denying the antecedent fallacy: ¬P doesn't prove ¬Q."
            },
            {
                "question": f"Given: \"{premise}\" We observe that {P}. Can we conclude that {Q}?",
                "expected_answer": "Yes",
                "reasoning": "Modus ponens: P→Q, P, therefore Q."
            }
        ]
        
        sets.append({
            "id": generate_id(),
            "category": "modus_tollens",
            "premise": premise,
            "questions": questions,
            "num_questions": len(questions),
            "logical_form": f"P→Q, where P=\"{P}\", Q=\"{Q}\""
        })
    
    return sets[:n]


# ============================================================
# CATEGORY 6: COMMONSENSE ENTAILMENT
# Tests: everyday logical entailments and their contrapositives
# ============================================================

def generate_commonsense_sets(n=85):
    """Generate commonsense entailment question sets."""
    
    # Curated commonsense entailment pairs
    # (action/event, entailed consequence, category)
    commonsense_rules = [
        ("John bought a new car", "John spent money", "financial"),
        ("Sarah ran a marathon", "Sarah exercised", "activity"),
        ("The vase fell off the table", "The vase was on the table before", "temporal"),
        ("Tom graduated from college", "Tom attended college", "prerequisite"),
        ("Lisa cooked pasta for dinner", "Lisa used a kitchen", "location"),
        ("The baby is crying", "The baby is awake", "state"),
        ("Mark drove to work", "Mark has a driver's license", "prerequisite"),
        ("Emily sent an email", "Emily used a computer or phone", "instrument"),
        ("The plant is growing", "The plant is alive", "state"),
        ("Robert won the chess tournament", "Robert played chess", "activity"),
        ("Anna translated the document from French", "Anna knows French", "prerequisite"),
        ("The ice cream melted", "The temperature was above freezing", "condition"),
        ("James unlocked the door", "James had a key or access method", "instrument"),
        ("The concert sold out", "People bought tickets", "consequence"),
        ("Sophie read the entire book", "Sophie can read", "prerequisite"),
        ("The river flooded", "There was excessive water", "cause"),
        ("Mike repaired the bicycle", "The bicycle was broken", "prerequisite"),
        ("The snow melted on the road", "The road temperature was above 0°C", "condition"),
        ("Clara painted a portrait", "Clara used paint", "instrument"),
        ("The fire alarm went off", "There was smoke or fire or a test", "cause"),
        ("Daniel passed the bar exam", "Daniel studied law", "prerequisite"),
        ("The restaurant received a health code violation", "An inspector visited", "prerequisite"),
        ("The satellite transmitted data", "The satellite was in orbit", "state"),
        ("Amy returned the library book", "Amy had borrowed a library book", "prerequisite"),
        ("The volcano erupted", "There was magma pressure underground", "cause"),
        ("Ben scored a goal", "Ben was playing a sport", "activity"),
        ("The bridge collapsed", "The bridge had structural weakness", "cause"),
        ("Rachel performed surgery", "Rachel is a trained medical professional", "prerequisite"),
        ("The phone battery died", "The phone was in use or not charged", "cause"),
        ("Kevin taught a class", "Kevin has knowledge of the subject", "prerequisite"),
        ("The tide went out", "The moon exerted gravitational pull", "cause"),
        ("Laura published a research paper", "Laura conducted research", "prerequisite"),
        ("The bread rose in the oven", "Yeast was active in the dough", "cause"),
        ("Chris flew to Tokyo", "Chris was at an airport", "location"),
        ("The stock price crashed", "Investors sold shares", "cause"),
        ("Nancy performed a piano recital", "Nancy can play piano", "prerequisite"),
        ("The experiment yielded positive results", "The experiment was conducted", "prerequisite"),
        ("Oliver caught a fish", "Oliver was near water", "location"),
        ("The lightbulb burned out", "The lightbulb was in use", "state"),
        ("Diana won a Nobel Prize", "Diana made a significant contribution to her field", "prerequisite"),
        ("The car ran out of gas", "The car had gas before", "temporal"),
        ("Peter submitted his tax return", "Peter earned income", "prerequisite"),
        ("The window fogged up", "There was a temperature difference between inside and outside", "cause"),
        ("Helen defended her PhD thesis", "Helen wrote a PhD thesis", "prerequisite"),
        ("The package was delivered", "Someone shipped the package", "prerequisite"),
        ("The soup boiled over", "The soup was being heated", "state"),
        ("Tim scored 100% on the exam", "Tim took the exam", "prerequisite"),
        ("The company went bankrupt", "The company existed before", "temporal"),
        ("Maria sang at the opera", "Maria can sing", "prerequisite"),
        ("The roof leaked during the storm", "The roof had damage or weakness", "cause"),
        ("George published a novel", "George wrote a novel", "prerequisite"),
        ("The dog fetched the ball", "Someone threw the ball", "prerequisite"),
        ("The server crashed", "The server was running", "state"),
        ("Sandra climbed Mount Kilimanjaro", "Sandra was physically capable of climbing", "prerequisite"),
        ("The election results were announced", "An election took place", "prerequisite"),
        ("The patient recovered from surgery", "The patient underwent surgery", "prerequisite"),
        ("The cake burned in the oven", "The cake was placed in the oven", "prerequisite"),
        ("Tony won an Olympic gold medal", "Tony competed in the Olympics", "activity"),
        ("The ship sank", "The ship was in water", "state"),
        ("Megan learned to speak Japanese", "Megan studied Japanese", "prerequisite"),
        ("The rocket launched successfully", "The rocket was on a launch pad", "state"),
        ("The courtroom judge issued a verdict", "A trial took place", "prerequisite"),
        ("The newspaper printed a retraction", "The newspaper previously published an error", "prerequisite"),
        ("William completed the puzzle", "William worked on the puzzle", "activity"),
        ("The power went out in the building", "The building had electrical power before", "temporal"),
        ("Jane received a promotion at work", "Jane was employed", "prerequisite"),
        ("The tree fell during the storm", "The tree was standing before", "temporal"),
        ("Alex translated the poem from Russian to English", "Alex knows both Russian and English", "prerequisite"),
        ("The butterfly emerged from the cocoon", "The butterfly was in a cocoon", "temporal"),
    ]
    
    sets = []
    used = set()
    
    for _ in range(n * 3):
        if len(sets) >= n:
            break
        
        rule = random.choice(commonsense_rules)
        event, entailment, cat = rule
        
        if event in used:
            continue
        used.add(event)
        
        questions = [
            {
                "question": f"Given that \"{event}\", is it reasonable to conclude that \"{entailment}\"?",
                "expected_answer": "Yes",
                "reasoning": f"Commonsense entailment ({cat}): {event} logically implies {entailment}."
            },
            {
                "question": f"If we know that it is NOT the case that \"{entailment.lower()}\", is it possible that \"{event.lower()}\"?",
                "expected_answer": "No",
                "reasoning": f"Contrapositive of commonsense entailment: if the entailment is false, the event couldn't have happened."
            },
            {
                "question": f"If we know that \"{entailment.lower()}\", can we conclude with certainty that \"{event.lower()}\"?",
                "expected_answer": "No, not with certainty",
                "reasoning": f"Affirming the consequent: the entailment being true doesn't uniquely determine this specific event."
            }
        ]
        
        sets.append({
            "id": generate_id(),
            "category": "commonsense_entailment",
            "premise": f"{event} → {entailment}",
            "questions": questions,
            "num_questions": len(questions),
            "logical_form": f"Event: \"{event}\" entails \"{entailment}\" ({cat})"
        })
    
    return sets[:n]


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_full_benchmark():
    """Generate the full ConsistencyBench benchmark."""
    print("Generating ConsistencyBench...")
    
    all_sets = []
    
    print("  Generating contrapositive sets...")
    contrapositive = generate_contrapositive_sets(85)
    all_sets.extend(contrapositive)
    print(f"    Generated {len(contrapositive)} sets")
    
    print("  Generating transitivity sets...")
    transitivity = generate_transitivity_sets(85)
    all_sets.extend(transitivity)
    print(f"    Generated {len(transitivity)} sets")
    
    print("  Generating syllogistic sets...")
    syllogistic = generate_syllogistic_sets(85)
    all_sets.extend(syllogistic)
    print(f"    Generated {len(syllogistic)} sets")
    
    print("  Generating negation sets...")
    negation = generate_negation_sets(85)
    all_sets.extend(negation)
    print(f"    Generated {len(negation)} sets")
    
    print("  Generating modus tollens sets...")
    modus_tollens = generate_modus_tollens_sets(85)
    all_sets.extend(modus_tollens)
    print(f"    Generated {len(modus_tollens)} sets")
    
    print("  Generating commonsense entailment sets...")
    commonsense = generate_commonsense_sets(85)
    all_sets.extend(commonsense)
    print(f"    Generated {len(commonsense)} sets")
    
    # Summary statistics
    total_questions = sum(s["num_questions"] for s in all_sets)
    print(f"\n=== ConsistencyBench Summary ===")
    print(f"Total question sets: {len(all_sets)}")
    print(f"Total individual questions: {total_questions}")
    print(f"Categories:")
    for cat in ["contrapositive", "transitivity", "syllogistic", "negation", "modus_tollens", "commonsense_entailment"]:
        cat_sets = [s for s in all_sets if s["category"] == cat]
        cat_qs = sum(s["num_questions"] for s in cat_sets)
        print(f"  {cat}: {len(cat_sets)} sets, {cat_qs} questions")
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "consistency_bench.json")
    with open(output_path, "w") as f:
        json.dump({
            "name": "ConsistencyBench",
            "version": "1.0",
            "description": "A benchmark for evaluating cross-query logical consistency in LLMs",
            "num_sets": len(all_sets),
            "num_questions": total_questions,
            "categories": ["contrapositive", "transitivity", "syllogistic", "negation", "modus_tollens", "commonsense_entailment"],
            "sets": all_sets
        }, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    return all_sets


if __name__ == "__main__":
    generate_full_benchmark()
