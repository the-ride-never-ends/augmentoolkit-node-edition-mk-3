
###############################
#### Augmentoolkit Prompts ####
###### By E.P. Armstrong ######
#### Unless otherwise noted ###
###############################

These prompts are for augmentoolkit's functions.

1): The prompts are, and must be, txt files named after the functions they go to.
    Variant prompts for each function's assistant mode are appended with "_assistant_mode". 
    Changing these without also changing the function to reflect the changes will break the function.

2): The prompt must contain the {f-string} arguments for the function they go to. 
    Changing these without also changing the function to reflect the changes will break the function.

3): A node may have multiple prompts associated with it, as nodes may use more than one function.

4): Multiple nodes may use the same prompt, as one function may be used by multiple nodes.




create_scenario_plan_many_tuples <- Under multi_turn_conversation
create_scenario_many_tuples <- Under make_multiturn_scenario Note: Multiturn to multi_turn for consistancy's sake.
create_character_card_plan_many_tuples <- Under make_multiturn_character
create_character_card_plan <- Under create_character_card_plan
create_character_card_plan_many_tuples <- Under make_multiturn_character
create_character_card_assistant_mode <- Under create_character_card

generate_questions <- Under generate_qa_tuples
check_question <- Under vet_question_loop
check_qa_tuple_context <- WHAT THE FUCK DOES THIS PROMPT GO TO????
multi_turn_conversation <- Under multi_turn_conversation
multi_turn_conversation_assistant_mode <- Under multi_turn_conversation
judge_paragraph <- Under judge_paragraph
generate_questions_plan <- Under generate_qa_tuples
generate_new_question <- Under vet_question_loop
check_answer_relevancy_with_text <- Under check_answer_relevancy_with_text
check_answer <- Under vet_answer_accuracy_loop
decision_prompt_judge_paragraphs <- Note: Rewrite this one


revise_qa_tuples
check_qatuple_context


###############################
#### NODE-PROMPT MAPPINGS #####
###############################

"Create a Simplified List Copy of the Dataset, then Process the Conversations"

"Create a Simplified List Copy of the Dataset"

"Make Dataset: Multi-turn Conversation (Simple)"

"Return Multi-turn Conversation Info"

"Ensure Multiple Answers are the Same"

	make_multiturn_conversation
		multi_turn_conversation
		create_scenario_plan_many_tuples


"Make Dataset (Multi-turn Conversation)"

"Return How Many Lines of Dialogue Were Generated"

"Generate QA Tuples (Advanced)"

generate_questions_plan
generate_questions

vet_question_loop

"Generate QA Tuples"
	Prompt: generate_questions
	Function: generate_qa_tuples
		Prompt: generate_questions_plan
	vet_question_loop
		check_question
		
		vet_answer_relevance_loop
			check_answer_relevancy_with_text
			generate_new_question
		vet_answer_accuracy_loop
			check_answer

"Group Revised QA Tuples by Paragraph"

"Revise QA Tuples"
	revise_qa_tuples
	check_qatuple_context

###############################
###### UNMAPPED-PROMPTS #######
###############################

These prompts are for experimental functions or functions still under testing.

llm_as_a_judge
llm_as_a_judge_assistant_mode

###############################
##### DEPRICATED PROMPTS ######
###############################

These prompts are for deprecated functions.
They remain in the folder for testing purposes or revival of old functions in updates



regenerate_answer_to_constrain_to_text <- Unmapped???
regenerate_answer <- Unmapped???
make_regenerate_question_plan <- Unmapped???
make_regenerate_answer_plan <- Unmapped???
make_regenerate_answer_constrain_to_text_plan <- Unmapped???
single_turn_conversation <- Unmapped???
single_turn_conversation_assistant_mode <- Unmapped???
ensure_multiple_answers_consistent <- Unmapped???
ensure_answer_consistent <- Unmapped???
generate_short_questions <- Unmapped???
regenerate_question_DEPRICATED <- Unmapped???
regenerate_question <- Unmapped???
create_thought_plan <- Unmapped???
create_scenario_plan <- Umapped???
create_scenario <- Umapped???
create_character_card <- Umapped???










