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
