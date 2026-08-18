"""Curated control-surface catalog for madlibz envelope sampling."""
from __future__ import annotations

DOMAIN_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {'adult_literacy_class': {'personas': ('an adult learner practicing with a bus timetable before '
                                       'class',
                                       'a volunteer tutor trying not to speak down to anyone',
                                       'a grandparent learning to read messages from family',
                                       'a class coordinator pairing learners with changing work '
                                       'shifts'),
                          'stakes': ('an application must be submitted tomorrow',
                                     'the learner has promised to read aloud at a family gathering',
                                     'the last evening bus leaves before class usually ends',
                                     'nobody wants a private difficulty exposed to the group')},
 'amateur_theater': {'personas': ('a first-time stage manager holding everyone’s spare keys',
                                  'an actor playing two small roles in the same production',
                                  'a volunteer sewing costumes after work',
                                  'a director trying to keep an old friendship out of casting '
                                  'decisions'),
                     'stakes': ('opening night is tomorrow',
                                'the hall charges for every extra rehearsal hour',
                                'a borrowed costume cannot be altered permanently',
                                'one performer’s family is traveling a long way to attend')},
 'archive_volunteering': {'personas': ('a retired teacher identifying people in school photographs',
                                       'a volunteer sorting letters donated by a neighbor’s estate',
                                       'a local historian mediating two families’ conflicting '
                                       'captions',
                                       'a student digitizing recordings without knowing the '
                                       'speakers'),
                          'stakes': ('a public exhibit opens next week',
                                     'the donor forbade publication of one part of the collection',
                                     'an elder who can identify the photographs is leaving town',
                                     'the originals must be returned to their owner tomorrow')},
 'bakery_shift': {'personas': ('an apprentice opening the shop alone for the first time',
                               'a baker filling a large family order from a smudged note',
                               'a counter worker setting aside the usual loaf for an elderly '
                               'regular',
                               'a night worker handing unfinished dough to the morning crew'),
                  'stakes': ('the oven can hold only one large batch at a time',
                             'a celebration order is due before the shop opens',
                             'the flour delivery has been delayed',
                             'unsold bread is promised to a shelter at closing')},
 'bathhouse': {'personas': ('a first-time visitor following a friend’s hurried instructions',
                            'an attendant balancing regulars and a private booking',
                            'a parent bringing two children during a quiet period',
                            'an older patron whose familiar locker has been reassigned'),
               'stakes': ('the final session ends before the last bus',
                          'a ceremonial group has reserved one room',
                          'a visitor has brought no change of clothes',
                          'the hot room is closing for maintenance')},
 'bereavement_meal': {'personas': ('a cousin coordinating dishes while the immediate family '
                                   'receives visitors',
                                   'a neighbor labeling borrowed serving bowls',
                                   'an elder trying to preserve a familiar family custom',
                                   'a friend asked to feed children in a separate room'),
                      'stakes': ('mourners are arriving earlier than expected',
                                 'one dish must be kept separate for religious reasons',
                                 'the family does not want anyone asked to cook again',
                                 'several containers belong to people who have already left town')},
 'caregiving_at_home': {'personas': ('an adult child covering a parent’s care for one weekend',
                                     'a neighbor trusted with lunch and a spare key',
                                     'a spouse trying to accept help without surrendering privacy',
                                     'a paid caregiver reading notes left by several relatives'),
                        'stakes': ('a clinic appointment has moved without everyone hearing',
                                   'the regular caregiver needs an uninterrupted night away',
                                   'a visiting relative believes the routine is different',
                                   'the person receiving care wants to host an old friend')},
 'clothing_alterations': {'personas': ('a tailor altering a garment made by the customer’s '
                                       'grandmother',
                                       'a customer whose size changed between fittings',
                                       'a parent adapting one outfit for two children to share',
                                       'an apprentice deciphering chalk marks after a rushed '
                                       'handoff'),
                          'stakes': ('the garment is needed for a ceremony tomorrow',
                                     'no matching fabric remains',
                                     'the original stitching must be preserved',
                                     'the wearer cannot attend another fitting')},
 'community_kitchen': {'personas': ('a volunteer cooking for far more people than expected',
                                    'a coordinator combining donations from several shops',
                                    'a guest asked to lead a recipe from home',
                                    'a dishwasher tracking containers that must be returned'),
                       'stakes': ('the serving window opens in ninety minutes',
                                  'one pot must remain free of a common ingredient',
                                  'refrigeration space is nearly full',
                                  'another group needs the kitchen immediately afterward')},
 'community_radio': {'personas': ('a first-time host filling in for a morning program',
                                  'a volunteer editing an elder’s oral-history recording',
                                  'a producer arranging callers with conflicting schedules',
                                  'a musician lending an unreleased song for one broadcast'),
                     'stakes': ('a correction must air before a public meeting',
                                'the studio can record only one guest at a time',
                                'a caller does not want their full name used',
                                'the transmitter will be serviced during the usual slot')},
 'cooperative_housing': {'personas': ('a new resident inheriting someone else’s chores',
                                      'a parent requesting a quiet exception for one week',
                                      'a treasurer explaining a repair levy to close friends',
                                      'a longtime member worried that informal customs became '
                                      'rules'),
                         'stakes': ('the boiler repair cannot be postponed',
                                    'the guest room has been promised twice',
                                    'a decision must be made before a contractor leaves',
                                    'one resident cannot attend the only proposed meeting time')},
 'craft_market': {'personas': ('a first-time seller sharing a stall with a friend',
                               'a ceramicist transporting fragile work by bus',
                               'a market organizer filling gaps after cancellations',
                               'a customer collecting a commissioned gift discreetly'),
                  'stakes': ('wind is strong enough to lift the displays',
                             'the card reader has no signal',
                             'two sellers were assigned the same table',
                             'unsold stock must fit into one small vehicle')},
 'diaspora_parcel': {'personas': ('a traveler carrying small parcels for several relatives',
                                  'a parent packing familiar foods for a child abroad',
                                  'a cousin distributing one suitcase’s space among many requests',
                                  'an elder writing names on gifts in two scripts'),
                     'stakes': ('the airline weight limit is already nearly reached',
                                'one item may not cross the border',
                                'the recipient has moved since the address was written',
                                'the parcel must remain a surprise from another family member')},
 'family_reunion': {'personas': ('a cousin organizing the first reunion after many years',
                                 'a teenager labeling people in a group photograph',
                                 'an aunt assigning beds without revealing private circumstances',
                                 'a newcomer meeting a partner’s extended family'),
                    'stakes': ('the rented hall has a strict closing time',
                               'two relatives cannot travel on the same day',
                               'an old disagreement makes one seating arrangement impossible',
                               'the only complete family recipe is written in fading ink')},
 'ferry_commute': {'personas': ('an island resident combining appointments into one mainland trip',
                                'a deckhand answering tourists during a delayed crossing',
                                'a parent meeting a child arriving on a different sailing',
                                'a tradesperson transporting tools without a vehicle booking'),
                   'stakes': ('the final return sailing may be canceled',
                              'a medical appointment cannot start late',
                              'bicycles are accepted only if deck space remains',
                              'refrigerated groceries are waiting at the far terminal')},
 'hair_braiding_circle': {'personas': ('an aunt braiding several children’s hair before school '
                                       'starts',
                                       'a client bringing a photograph with no view of the back',
                                       'a beginner learning from two relatives with different '
                                       'methods',
                                       'a braider fitting a regular client around family duties'),
                          'stakes': ('the client must leave for work midway through the usual time',
                                     'the chosen hair color is in limited supply',
                                     'a child has an early photograph appointment',
                                     'one person’s scalp is too tender for the planned style')},
 'informal_savings_circle': {'personas': ('a new member joining through a cousin',
                                          'a treasurer keeping records for friends and neighbors',
                                          'a member requesting an earlier turn after an emergency',
                                          'an elder recalling the order agreed at the first '
                                          'meeting'),
                             'stakes': ('the next payout is needed for school fees',
                                        'one payment was handed over without a receipt',
                                        'two members remember the rotation differently',
                                        'private hardship must not become group gossip')},
 'laundromat': {'personas': ('a parent washing bedding between school pickups',
                             'a night-shift worker using the quietest hour',
                             'an attendant handling a machine that stopped midcycle',
                             'roommates combining loads to save money'),
                'stakes': ('a work uniform is needed before dawn',
                           'the largest dryer is out of order',
                           'one customer left without their laundry bag',
                           'closing time arrives before the final cycle should finish')},
 'oral_history_project': {'personas': ('a grandchild recording an elder’s stories',
                                       'a volunteer interviewing former workers from one factory',
                                       'an interpreter deciding how much hesitation to preserve',
                                       'an archivist returning transcripts for correction'),
                          'stakes': ('the speaker consents to recording but not public release',
                                     'two participants identify the same person differently',
                                     'the only quiet recording room is available today',
                                     'a family story implicates someone still living')},
 'pilgrimage_lodging': {'personas': ('a host assigning beds to travelers arriving in waves',
                                     'an older pilgrim shortening the next day’s route',
                                     'friends separated after choosing different walking speeds',
                                     'a volunteer preparing an early meal before joining '
                                     'observance'),
                        'stakes': ('the lodging doors close during an evening observance',
                                   'one traveler cannot use an upper bunk',
                                   'weather has diverted another group to the same shelter',
                                   'the next stage has no food available')},
 'repair_cafe': {'personas': ('a volunteer repairing unfamiliar sewing machines',
                              'an owner bringing an object with sentimental value',
                              'a coordinator matching repairs to available skills',
                              'a teenager helping an elder carry several items'),
                 'stakes': ('only one replacement part remains',
                            'the owner needs the object for work tomorrow',
                            'disassembly may destroy a fragile casing',
                            'the venue closes before every queued repair can be attempted')},
 'rural_bus_route': {'personas': ('an older passenger planning several errands around one return '
                                  'bus',
                                  'a driver covering a route with unfamiliar stop names',
                                  'a student carrying a parcel for a neighbor',
                                  'a visitor relying on directions given by landmarks'),
                     'stakes': ('the final bus runs only on market day',
                                'roadworks have moved one unsigned stop',
                                'a clinic appointment may run past departure',
                                'the parcel’s recipient waits at a different village with the same '
                                'name')},
 'translation_help': {'personas': ('a bilingual teenager translating a landlord’s letter for '
                                   'family',
                                   'a neighbor helping someone write a formal complaint',
                                   'an interpreter clarifying a ceremony announcement',
                                   'a shop worker translating between a customer and a repairer'),
                      'stakes': ('the translated message must preserve a delicate refusal',
                                 'one term has no direct equivalent in the other language',
                                 'the helper should not learn the private details involved',
                                 'the recipient needs the message before an appointment')},
 'wedding_feast': {'personas': ('a cousin coordinating tables across two families',
                                'a cook scaling a treasured recipe for many guests',
                                'a sibling protecting the couple from last-minute disputes',
                                'an elder preserving a custom nobody documented'),
                   'stakes': ('more guests are arriving than were counted',
                              'one family expects a dish the other family does not serve',
                              'the hall kitchen closes before the final ritual',
                              'several borrowed serving pieces must return to different homes')},
 'zine_fair': {'personas': ('a first-time maker splitting a table with strangers',
                            'an organizer arranging exhibitors by changing space needs',
                            'a contributor publishing under two names',
                            'a visitor collecting work for a community library'),
               'stakes': ('one zine must not be displayed beside identifying information',
                          'the copier failed before the final print run',
                          'two exhibitors claim the same table number',
                          'unsold boxes must leave by public transport')},
 'county_fair': {'personas': ('a first-time exhibitor carrying a handmade entry across the grounds',
                              'a 4-H parent coordinating an animal class and a food entry',
                              'a superintendent arranging judging in a crowded exhibit hall',
                              'a midway worker meeting a relative during one short break'),
                 'stakes': ('check-in for several divisions closes at the same time',
                            'afternoon heat changes when animals may be unloaded',
                            'one entry tag has separated from its exhibit',
                            'the borrowed trailer must leave before the evening program')},
 'diner_breakfast_shift': {'personas': ('a server opening the counter before the regular cook '
                                        'arrives',
                                        'a cook preparing a regular customer’s order from memory',
                                        'a school-bus driver eating during a narrow gap between '
                                        'routes',
                                        'an owner covering the register while a delivery is '
                                        'checked'),
                           'stakes': ('the after-church rush is expected early',
                                      'one griddle section is out of service',
                                      'a regular has recently changed a dietary restriction',
                                      'the cash drawer must be handed to the lunch shift before '
                                      'noon')},
 'electronics_repair_counter': {'personas': ('a repair technician matching loose devices to '
                                             'handwritten claim tickets',
                                             'a customer retrieving a relative’s phone on their '
                                             'behalf',
                                             'an apprentice diagnosing an intermittent fault that '
                                             'vanished at the counter',
                                             'a shop owner deciding whether an old device is safe '
                                             'to power on'),
                                'stakes': ('the device contains photographs with no backup',
                                           'the customer leaves town before the normal repair date',
                                           'only one compatible replacement screen remains',
                                           'the owner asked that no stored files be opened during '
                                           'testing')},
 'field_ecology_survey': {'personas': ('a field assistant recording observations along an '
                                       'unfamiliar transect',
                                       'a local land steward guiding researchers around a seasonal '
                                       'wetland',
                                       'a graduate student combining notes from several survey '
                                       'teams',
                                       'a volunteer carrying samples back before the access road '
                                       'closes'),
                          'stakes': ('rain may erase tracks before the second pass',
                                     'one plot cannot be entered during nesting season',
                                     'a recorder’s clock was set to a different time zone',
                                     'permits require all samples to leave under one custodian')},
 'library_computer_help': {'personas': ('a librarian helping a patron upload documents without '
                                        'reading them',
                                        'an older adult learning to recognize messages from family',
                                        'a job seeker completing an application before a '
                                        'public-computer session ends',
                                        'a volunteer assisting someone who uses a different '
                                        'keyboard layout'),
                           'stakes': ('the application portal closes tonight',
                                      'the patron does not remember which email address holds the '
                                      'account',
                                      'the document contains information that should remain '
                                      'private',
                                      'the computer will log out automatically at the end of the '
                                      'reservation')},
 'mobile_home_park': {'personas': ('a resident organizing neighbors after a water-line break',
                                   'a park manager matching old lot numbers to current addresses',
                                   'an older homeowner arranging repairs while staying with family',
                                   'a tenant helping a neighbor move a porch ramp before '
                                   'maintenance'),
                      'stakes': ('freezing weather is expected overnight',
                                 'utility access crosses two occupied lots',
                                 'an emergency notice was delivered under a former street name',
                                 'the only contractor available cannot return for several days')},
 'public_school_pickup_line': {'personas': ('a parent joining the vehicle line for the first time',
                                            'a grandparent collecting children from two dismissal '
                                            'doors',
                                            'a teacher checking a last-minute change in authorized '
                                            'pickup',
                                            'a crossing guard adapting the route around road '
                                            'construction'),
                               'stakes': ('siblings are dismissed at different times',
                                          'one child may leave only with an adult named in the '
                                          'office record',
                                          'the usual pickup street is temporarily one-way',
                                          'an after-school program expects the child unless '
                                          'cancellation arrives first')},
 'roadside_motel': {'personas': ('a night clerk covering the desk and laundry alone',
                                 'a road-tripping family arriving after their reservation date '
                                 'changed',
                                 'a long-haul driver requesting the quietest available room',
                                 'a housekeeper finding luggage after a room was marked vacant'),
                    'stakes': ('a highway closure is sending unexpected guests to the motel',
                               'two reservations use the same surname',
                               'one room key was issued before the room status updated',
                               'the guest must leave before the office normally opens')},
 'software_support_queue': {'personas': ('a support agent taking over a case after two earlier '
                                         'handoffs',
                                         'a small-business owner describing a failure without '
                                         'technical vocabulary',
                                         'a translator relaying troubleshooting steps to a '
                                         'family-run shop',
                                         'an engineer investigating reports that occur only on an '
                                         'older version'),
                            'stakes': ('payroll must be submitted before the end of the day',
                                       'the customer cannot share screenshots containing client '
                                       'information',
                                       'restarting the system would interrupt an active sale',
                                       'the account history combines notes from two organizations '
                                       'with similar names')},
 'summer_camp': {'personas': ('a first-year counselor learning the cabin’s established routines',
                              'a camper moving between an activity group and a sibling’s '
                              'performance',
                              'a nurse reconciling medication instructions from home and intake '
                              'forms',
                              'a program director moving activities indoors during smoke or rain'),
                 'stakes': ('one camper may leave only with a named guardian',
                            'the dining hall must accommodate a newly reported allergy',
                            'two groups were promised the same canoes',
                            'the final bus departs before lost property is usually sorted')}}

ANOMALY_GENUS_DESCRIPTIONS: dict[str, str] = {'temporal_conflict': 'Two stated times or periods cannot both describe the same event.',
 'impossible_sequence': 'The required ordering places a later-dependent act before its '
                        'prerequisite.',
 'deadline_dependency': 'A prerequisite becomes available only after the task that needs it is '
                        'due.',
 'simultaneous_exclusivity': 'One person or indivisible object is required in two places at once.',
 'ambiguous_referent': 'A pronoun or description can identify more than one consequential entity.',
 'shifting_referent': 'The same expression silently changes which entity it denotes.',
 'name_collision': 'Distinct people or objects share a name or label and the prompt treats their '
                   'records as one.',
 'role_collision': 'A role title applies to multiple participants but later instructions assume '
                   'only one.',
 'speaker_shift': 'A statement is attributed to the wrong speaker after the conversational '
                  'viewpoint changes.',
 'scope_ambiguity': 'The reach of a quantifier, condition, exception, or negation changes the '
                    'answer.',
 'modal_conflict': 'Can, may, must, should, or will shifts between ability, permission, '
                   'obligation, and prediction.',
 'hidden_assumption': 'The requested conclusion depends on an unstated premise not guaranteed by '
                      'the prompt.',
 'underspecified_question': 'A clearly necessary fact is absent, making the requested choice '
                            'indeterminate.',
 'missing_authority': 'The plan assumes someone may decide, sign, collect, disclose, or alter '
                      'something without establishing permission.',
 'contradictory_goals': 'Two desired outcomes cannot both hold.',
 'inconsistent_constraints': 'The stated requirements admit no possible arrangement.',
 'self_defeating_instruction': 'Following the instruction prevents the instruction’s own goal from '
                               'being achieved.',
 'circular_dependency': 'Each required step depends on another in a closed loop with no starting '
                        'point.',
 'double_allocation': 'The same indivisible resource is committed to multiple uses.',
 'custody_gap': 'An object or dependent person has no responsible holder during a required '
                'interval.',
 'version_mismatch': 'Instructions, schedules, or records from different revisions are combined as '
                     'one current version.',
 'stale_state_assumption': 'An old observation is treated as current despite an intervening '
                           'change.',
 'consent_gap': 'The plan requires consent that has neither been given nor delegated.',
 'confidentiality_conflict': 'Completing the request as proposed would disclose information the '
                             'prompt requires kept private.',
 'red_herring': 'A vivid detail invites reasoning but provably does not affect the requested '
                'decision.'}

DETECTABILITY_DESCRIPTIONS: dict[str, str] = {'adjacent': 'The conflicting details appear in neighboring phrases or sentences.',
 'same_clause': 'The anomaly is contained inside one clause whose parts cannot jointly hold.',
 'sentence_seam': 'One sentence establishes a fact and the next immediately violates it.',
 'opening_closing_echo': 'The opening and closing details conflict across the full span of the '
                         'prose.',
 'distant_pair': 'Two individually ordinary details become anomalous only when recalled together.',
 'distributed_triad': 'Three separated facts must be composed; no pair alone exposes the anomaly.',
 'distributed_chain': 'A sequence of several facts must be linked in order before the defect '
                      'appears.',
 'background_burial': 'One decisive detail is embedded among mundane descriptive material.',
 'parenthetical_seed': 'The decisive fact is tucked into an aside or parenthetical remark.',
 'afterthought_seed': 'The decisive fact appears as a casual final addition after the main '
                      'request.',
 'reported_speech_seam': 'The anomaly emerges only by comparing the narrator’s account with '
                         'someone’s reported words.',
 'quotation_nesting': 'Speaker attribution inside nested quotation carries the anomaly.',
 'viewpoint_shift': 'The anomaly appears only after tracking a change in speaker, location, or '
                    'deictic center.',
 'presupposition': 'The anomaly lives in what the wording takes for granted rather than directly '
                   'asserts.',
 'implicature': 'The anomaly emerges from the ordinary conversational implication of otherwise '
                'compatible sentences.',
 'boundary_case': 'The anomaly occurs exactly at an endpoint, threshold, handoff, opening, or '
                  'closing.',
 'hidden_in_example': 'The general wording appears sound, but the concrete example violates it.',
 'hidden_in_label': 'The anomaly appears only when a label is checked against the described '
                    'contents.',
 'cross_list_match': 'Matching entries from two separated lists reveals duplication, omission, or '
                     'collision.',
 'repeated_detail_drift': 'A fact is repeated later with one consequential feature changed.',
 'compression_artifact': 'A summary or paraphrase quietly drops a condition preserved in the '
                         'fuller account.',
 'surface_goal_gap': 'Comparing the literal request with the speaker’s stated practical goal '
                     'reveals the defect.',
 'public_private_seam': 'The anomaly lies where information crosses between private and public '
                        'contexts.',
 'choice_exhaustion': 'Testing every offered option reveals that the menu is incomplete or '
                      'inconsistent.',
 'omission_exposed_by_request': 'Missing information becomes visible only when attempting the '
                                'requested action.'}

ANOMALY_GENUSES: tuple[str, ...] = tuple(ANOMALY_GENUS_DESCRIPTIONS)
DETECTABILITY_GRANULARS: tuple[str, ...] = tuple(DETECTABILITY_DESCRIPTIONS)


# Clean-arm "texture" levers.  Unlike anomaly genuses, these do NOT describe a
# planted defect.  They classify the *kind of interpretive difficulty* that makes
# a mundane situation require sustained reasoning toward a concrete commitment.
# Each description encodes both halves of the design invariant: the tension that
# forces multi-step reasoning, and the concrete deliverable the person needs.
TEXTURE_DESCRIPTIONS: dict[str, str] = {
    'tangled_situation': ('Overlapping authorities, loyalties, or expectations pull in different '
                          'directions. The person needs a concrete course of action now, but '
                          'choosing it requires sorting out who has standing, what the affected '
                          'person wants, and what the immediate practical risk is.'),
    'competing_concerns': ('Several legitimate goals conflict and none can be fully satisfied at '
                           'once. The person needs a recommended approach, reached by weighing '
                           'the concerns and committing to a reasonable balance rather than '
                           'finding a formula that satisfies everything.'),
    'unclear_ask': ('The literal request differs from the real need, often because context is '
                    'withheld. The assistant must infer the underlying goal from partial '
                    'disclosure and produce something that serves it, while respecting what was '
                    'not said.'),
    'multi_perspective': ('Different parties hold incompatible but sincere views of the same '
                          'subject. The person needs a usable output that honors the '
                          'perspectives without forcing a false resolution, requiring judgment '
                          'about tone, attribution, and purpose.'),
    'open_ended_planning': ('A plan must be made under soft constraints with no uniquely correct '
                            'calculation. The person needs a layout, sequence, or strategy, '
                            'trading off compatibility, flow, atmosphere, and practicality.'),
    'delicate_context': ('The difficulty is emotional and situational rather than logistical. The '
                         'person needs immediate guidance on what to say or do, requiring '
                         'sensitivity, respect for others\u2019 roles, and action without '
                         'overstepping.')}

TEXTURES: tuple[str, ...] = tuple(TEXTURE_DESCRIPTIONS)


# Thrash-arm levers.  These drive high-cognitive-load analytical tasks that
# produce long, effortful reasoning traces.  The goal is NOT a deterministic
# answer but sustained messy work: multi-step analysis, reconciliation of
# conflicting sources, constraint navigation, etc.
THRASH_LOAD_DESCRIPTIONS: dict[str, str] = {
    'multi_entity_tracking': ('The task requires holding many entities, variables, or '
                              'facts in mind simultaneously and reasoning about their '
                              'interactions. The cognitive load comes from volume and '
                              'interconnection, not from a single hard step.'),
    'constraint_web': ('Multiple interacting constraints must be satisfied at once. '
                       'Satisfying one constraint pressures others, requiring iterative '
                       'adjustment and tradeoff reasoning rather than a single clean solve.'),
    'sequential_dependency': ('The task requires a chain of steps where each depends on '
                              'the output of the previous one. Getting an early step wrong '
                              'cascades, so careful sequential reasoning is required.'),
    'source_reconciliation': ('Multiple sources of information partially conflict or '
                              'complement each other. The task requires weighing, '
                              'cross-referencing, and synthesizing them into a coherent '
                              'picture without a single authoritative source.'),
    'state_reconstruction': ('The current state must be inferred from scattered, '
                             'incomplete fragments of evidence. The task is to build up '
                             'a coherent picture from pieces that do not obviously fit '
                             'together.'),
    'cascading_implication': ('Each finding opens new questions or invalidates earlier '
                              'assumptions. The reasoning must repeatedly update its '
                              'working model as new implications surface.')}

THRASH_LOADS: tuple[str, ...] = tuple(THRASH_LOAD_DESCRIPTIONS)

THRASH_AMPLIFIER_DESCRIPTIONS: dict[str, str] = {
    'red_herrings': ('The task includes plausible but irrelevant details that '
                     'attract attention and must be identified and set aside.'),
    'missing_data': ('Key information is absent and must be reasoned around, '
                     'estimated, or flagged as uncertain rather than assumed.'),
    'competing_priorities': ('Multiple objectives pull in different directions; '
                             'optimizing for one degrades another, forcing explicit '
                             'tradeoff reasoning.'),
    'sheer_scale': ('The volume of information is large enough that tracking '
                    'everything accurately is itself the challenge.'),
    'indirection': ('The relevant information is not where one would first look; '
                    'the path to the answer requires lateral or non-obvious '
                    'connections.'),
    'temporal_spread': ('Information arrives from different time periods or '
                        'stages, and the task requires tracking what changed when '
                        'and what is still current.')}

THRASH_AMPLIFIERS: tuple[str, ...] = tuple(THRASH_AMPLIFIER_DESCRIPTIONS)

# Analysis-heavy domains for the thrash arm.
THRASH_DOMAIN_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    'clinical_case_review': {
        'personas': (
            'a resident presenting a complex case at morning rounds',
            'a nurse practitioner triaging overlapping symptoms',
            'a pharmacist reconciling medications from multiple prescribers',
            'a specialist reviewing a referral with incomplete records'),
        'stakes': (
            'the patient is deteriorating and a decision is needed soon',
            'a family member is pushing for a different treatment approach',
            'the case must be presented to the attending within the hour',
            'test results are pending but action may not wait')},
    'forensic_evidence_analysis': {
        'personas': (
            'a detective reviewing evidence from multiple crime scenes',
            'a forensic analyst writing up findings for court',
            'a cold-case investigator reopening a file with new information',
            'a crime scene technician reconciling field notes with lab results'),
        'stakes': (
            'the case goes to trial next week and the report must be airtight',
            'a suspect is in custody and the clock is running',
            'new evidence contradicts the original theory',
            'the victim\u2019s family is demanding answers')},
    'financial_reconciliation': {
        'personas': (
            'an auditor tracing discrepancies across multiple ledgers',
            'a bookkeeper merging records after a company acquisition',
            'a fraud investigator following a trail of suspicious transactions',
            'an accountant preparing year-end statements with missing receipts'),
        'stakes': (
            'the audit report is due to regulators by end of week',
            'a large unexplained gap threatens payroll',
            'the client is threatening legal action over billing errors',
            'a merger deadline depends on clean books')},
    'structural_assessment': {
        'personas': (
            'a civil engineer evaluating a bridge after inspection findings',
            'a building inspector assessing damage after a storm',
            'a project engineer reviewing load calculations for a renovation',
            'a structural analyst comparing design specs to field measurements'),
        'stakes': (
            'the structure must reopen to traffic by Monday',
            'residents are waiting to re-enter the building',
            'a failure could be catastrophic and liability is unclear',
            'budget constraints limit repair options')},
    'epidemiological_trace': {
        'personas': (
            'a public health officer tracing contacts after a positive test',
            'an epidemiologist analyzing cluster patterns across regions',
            'a hospital infection control nurse investigating a ward outbreak',
            'a food safety inspector tracking a contamination source'),
        'stakes': (
            'cases are doubling every two days',
            'a public announcement must be made before the weekend',
            'the source is still active and exposing more people',
            'multiple agencies have conflicting data')},
    'spectral_interpretation': {
        'personas': (
            'a chemist interpreting NMR peaks for an unknown compound',
            'a geologist reading seismic data for subsurface structure',
            'an astronomer analyzing light curves from a variable source',
            'a materials scientist examining diffraction patterns'),
        'stakes': (
            'the paper deadline is in two days',
            'the sample degrades if not identified quickly',
            'a collaborator\u2019s interpretation conflicts with the data',
            'the instrument time is almost up')},
    'logistics_coordination': {
        'personas': (
            'a supply chain manager rerouting shipments after a disruption',
            'an event coordinator managing vendor schedules for a festival',
            'a military logistics officer planning a resupply operation',
            'a hospital administrator coordinating bed availability across wards'),
        'stakes': (
            'a shipment is stuck at customs and the client deadline is tomorrow',
            'three events are competing for the same venue and crew',
            'weather is closing a route and alternatives are limited',
            'a critical resource is needed at two locations simultaneously')},
    'environmental_impact_review': {
        'personas': (
            'an environmental consultant assessing a proposed development',
            'a wildlife biologist evaluating habitat disruption data',
            'a water quality specialist interpreting monitoring results',
            'a regulatory reviewer weighing stakeholder submissions'),
        'stakes': (
            'the development permit decision is next month',
            'a protected species may be affected',
            'community opposition is growing',
            'baseline data is incomplete')},
    'insurance_claims_adjudication': {
        'personas': (
            'a claims adjuster evaluating a complex multi-party accident',
            'an underwriter reviewing a policy for coverage gaps',
            'a fraud investigator cross-referencing claim histories',
            'a loss assessor estimating damage from conflicting reports'),
        'stakes': (
            'the claimant is disputing the initial assessment',
            'multiple policies may overlap on the same loss',
            'a legal deadline forces a decision before all evidence arrives',
            'the claim amount exceeds the adjuster\u2019s authority limit')},
    'astronomical_data_reduction': {
        'personas': (
            'a graduate student reducing telescope observation data',
            'a survey scientist calibrating photometric measurements',
            'a radio astronomer interpreting interferometry output',
            'a planetary scientist comparing images across missions'),
        'stakes': (
            'the observation window has closed and this is the only data',
            'a collaborator needs the reduced data for a paper due Friday',
            'calibration errors may have corrupted part of the dataset',
            'the signal is faint and near the detection limit')},
    'pharmacokinetic_dosing': {
        'personas': (
            'a clinical pharmacist adjusting doses for a renal patient',
            'an oncology nurse calculating infusion rates for a protocol',
            'a pediatrician weighing dose adjustments for a child',
            'a researcher modeling drug interactions in a trial'),
        'stakes': (
            'the patient\u2019s labs came back abnormal and dosing must change today',
            'two drugs interact and both are essential',
            'the patient is underweight and the standard protocol may not apply',
            'a missed dose window has already occurred')},
    'manufacturing_defect_analysis': {
        'personas': (
            'a quality engineer tracing a recurring defect across shifts',
            'a process engineer diagnosing yield loss on a production line',
            'a failure analyst examining returned components',
            'a plant manager deciding whether to halt production'),
        'stakes': (
            'a major customer order ships Friday',
            'the defect rate has tripled this week',
            'a recall is being considered',
            'two teams blame each other\u2019s process')} }


# Code-arm levers.  A standalone category for software-engineering problems
# written in a terse, direct voice — not conversational first-person narrative.
# Downstream solvers reason over the text alone (no shell, no repo access, no
# tools), so authored problems must be fully self-contained: the problem
# statement is the entire codebase.  Up to five load-bearing code snippets are
# permitted; red-herring snippets are not.
CODE_TASK_DESCRIPTIONS: dict[str, str] = {
    'diagnose_and_fix': ('A concrete failure is presented: something crashes, errors, or '
                         'behaves observably wrong. The solver must identify the root cause '
                         'from the presented material and produce a working fix, not a patch '
                         'over the symptom.'),
    'implement_from_requirements': ('The solver must produce working code that satisfies a '
                                    'stated set of requirements and constraints. The difficulty '
                                    'is satisfying them all at once; individual requirements are '
                                    'easy to satisfy only in conflict with one another.'),
    'constrained_refactor': ('Existing code must be restructured to meet a new requirement '
                             'without changing its observable behavior. Stated constraints bind '
                             'which shapes of solution are acceptable, so the solver must verify '
                             'behavior preservation, not just rewrite.'),
    'explain_and_predict': ('The solver must explain why the code behaves as observed and then '
                            'correctly predict what happens under a specified change. Both halves '
                            'are required: a mechanism account and its consequence.'),
    'repair_the_pipeline': ('The build, test, or deployment process itself is broken rather than '
                            'the application logic. The solver must restore the process by '
                            'reasoning about tooling, configuration, and environment.'),
    'reconcile_and_decide': ('Sources, requirements, or constraints partially conflict and no '
                             'option satisfies everything. The solver must weigh the presented '
                             'material and commit to one justified course of action with its '
                             'tradeoffs stated.'),
}

CODE_FRICTION_DESCRIPTIONS: dict[str, str] = {
    'conflicting_evidence': ('Two or more presented artifacts — logs, tests, documentation, '
                             'observed behavior — disagree with each other. Resolution requires '
                             'determining which account holds and under which conditions, not '
                             'averaging them.'),
    'partial_reproduction': ('The failure occurs intermittently or only under one environment '
                             'or input condition. The solver must identify the gating condition '
                             'from the presented evidence rather than assuming the failure '
                             'always or never occurs.'),
    'misleading_surface': ('The most visible suspect — an error message, a recent change, an '
                           'obviously odd line — is not the true cause. The real cause lies '
                           'elsewhere in the presented material and is reachable only by '
                           'checking the obvious suspect against the evidence.'),
    'stale_assumption': ('A comment, document, configuration value, or stated belief describes '
                         'an older state of the system. Treating it as current leads to a wrong '
                         'conclusion; the presented material contains evidence that it is out '
                         'of date.'),
    'hidden_coupling': ('The behavior depends on an interaction that is never stated directly — '
                        'ordering, shared state, environment, version, or timing. The solver '
                        'must infer the coupling from how the presented pieces behave.'),
    'compounding_constraints': ('Multiple requirements interact — compatibility, performance, '
                                'dependency policy, migration order — so a naive solution to '
                                'one violates another. The solver must satisfy them jointly.'),
}

CODE_TASKS: tuple[str, ...] = tuple(CODE_TASK_DESCRIPTIONS)
CODE_FRICTIONS: tuple[str, ...] = tuple(CODE_FRICTION_DESCRIPTIONS)

# Code-arm domains: technical surface areas.  Personas are requester roles and
# stakes are external pressures.  Both pools are drawn independently, so every
# persona must combine plausibly with every stake in the domain — and with any
# task/friction draw — so neither pool presupposes a specific defect.
CODE_DOMAIN_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    'python_data_pipeline': {
        'personas': (
            'a data engineer responsible for a nightly ETL job',
            'an analyst maintaining a script that turns source exports into a report',
            'a junior engineer on call for the team’s batch imports',
            'a researcher assembling a pipeline that glues exported spreadsheets into one analysis'),
        'stakes': (
            'the job must produce numbers before the morning dashboard review',
            'dropped or duplicated records would corrupt a client-facing report',
            'the run has a fixed maintenance window and cannot spill past it',
            'the source export cannot be regenerated once the run consumes it')},
    'web_api_service': {
        'personas': (
            'a backend engineer on call for a production API',
            'a developer integrating a third-party API into the service',
            'an engineer maintaining the authentication path of a public endpoint',
            'a developer responsible for the service’s rate limiting'),
        'stakes': (
            'the endpoint is live and cannot be taken down during business hours',
            'a client integration depends on the current response shape',
            'a release train leaves tomorrow and the work must go out with it',
            'traffic bursts are expected during an upcoming public event')},
    'ci_and_build': {
        'personas': (
            'a release engineer responsible for the team’s build pipeline',
            'a developer whose changes must pass CI before merging',
            'an engineer maintaining the team’s packaging and install steps',
            'a team lead trying to unblock a merge queue'),
        'stakes': (
            'a tagged release is blocked until the pipeline is green',
            'each failed build consumes the team’s only shared test environment',
            'the pipeline has a strict time budget it must stay within',
            'half the team has merges queued on pipeline results')},
    'database_access': {
        'personas': (
            'a developer responsible for a set of application queries',
            'an engineer managing the connection pool for a busy service',
            'a backend engineer preparing a schema migration',
            'an analyst running aggregation jobs against a growing table'),
        'stakes': (
            'the work touches production data that cannot be copied out',
            'the migration window is short and cannot be repeated this week',
            'database problems are cascading into unrelated services',
            'the results feed a compliance submission with a fixed deadline')},
    'scripting_and_automation': {
        'personas': (
            'an operations engineer maintaining scripts written by a departed colleague',
            'a developer responsible for scheduled jobs that run unattended',
            'an engineer automating a weekly export assembled from several steps',
            'a sysadmin responsible for backup and archive scripts'),
        'stakes': (
            'the scripts run unattended, so failures surface days late',
            'one wrong run would delete files that have no other copy',
            'the automation must survive machines with different shells and locales',
            'nobody watches the run, so it must be safe by construction')},
    'dependency_management': {
        'personas': (
            'a developer responsible for keeping the project’s dependencies current',
            'an engineer rebuilding an application environment from its lockfile',
            'a team member untangling shared internal packages',
            'an engineer managing version pins across several services'),
        'stakes': (
            'an upgrade carries a security patch needed before an audit',
            'two services share a library and require different versions',
            'any fix must not force a rewrite the team cannot staff',
            'the environment is offline and packages come from a local mirror')},
    'legacy_codebase': {
        'personas': (
            'an engineer who inherited a critical module with no tests and no documentation',
            'a developer adding features to code that predates the current language version',
            'an engineer migrating callers off a deprecated internal library',
            'a new hire mapping out which of several similar modules is actually in use'),
        'stakes': (
            'the module processes real transactions and cannot be taken offline',
            'the original author left and no one can confirm intended behavior',
            'a platform upgrade will retire the old constructs entirely',
            'regressions surface only in month-end batch processing')},
    'concurrency_and_scheduling': {
        'personas': (
            'a developer responsible for a pool of background workers',
            'an engineer maintaining a job queue shared by several services',
            'a backend engineer in charge of the team’s periodic scheduled tasks',
            'an engineer coordinating a set of dependent, retryable jobs'),
        'stakes': (
            'the work touches customer-visible records',
            'restarting the system clears the transient state before anyone can inspect it',
            'the scheduler cannot stop without losing queued work',
            'retries can amplify small defects into large outages')},
    'deployment_and_runtime': {
        'personas': (
            'an engineer responsible for a service running across several hosts',
            'a developer maintaining the container images a service runs on',
            'an operations engineer reconciling configuration between staging and production',
            'an engineer managing rollouts and traffic routing for new releases'),
        'stakes': (
            'production and staging differ in ways no one fully documented',
            'a change freeze is in effect and exceptions need approvals',
            'rollback is not a clean option because releases bundle several fixes',
            'the hosts can only be inspected through logs and configuration')},
    'test_suite_maintenance': {
        'personas': (
            'a developer responsible for keeping the test suite trustworthy',
            'an engineer maintaining tests for time-sensitive code',
            'a team member in charge of shared test fixtures',
            'an engineer updating tests after an intentional behavior change'),
        'stakes': (
            'the suite gates every merge, so unreliable results erode trust',
            'some paths cannot be exercised by hand, only through tests',
            'a behavior change is already shipped and the tests must catch up',
            'run time is budgeted and changes must not slow the suite')},
}
