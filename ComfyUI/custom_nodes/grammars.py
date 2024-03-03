import os
from llama_cpp import LlamaGrammar

class Grammars:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_grammars": (["True", "False"],), # Currently does nothing.
            },
            "hidden": {},
        }
    RETURN_TYPES = ()
    FUNCTION = "get_grammars"

    output_node = True

    CATEGORY = "output_validation"

    def get_grammars(use_grammars): #Placeholder function so the program doesn't print all the grammars during the start-up sequence.
    # NOTE might struggle with very complex answers that have more than nine parts to them. This can be amended by adding more options to the "compare-step" rule, or making a more general pattern, if your use-case requires it.
        if use_grammars == "True":
            pass
        else:
            pass

        answer_accurate_grammar = LlamaGrammar.from_string(
            r"""                     
        
           
        # Root rule specifying the overall structure of the analysis
        root ::= text-analysis answer-breakdown accuracy-check final-judgment

        # Text analysis
        text-analysis ::= "### Text Analysis:" "\n" identify-key-info categorize-info-type

        identify-key-info ::= "#### Identify Key Information: " text-info-detail "\n"
        categorize-info-type ::= "#### Categorize Information Type: " info-type-detail "\n\n"

        # Answer breakdown
        answer-breakdown ::= "### Answer Breakdown:" "\n" dissect-answer identify-answer-type

        dissect-answer ::= "#### Dissect the Answer: " answer-detail "\n"
        identify-answer-type ::= "#### Identify Answer Type: " answer-type-detail "\n\n"

        # Accuracy check
        accuracy-check ::= "### Accuracy Check:" "\n" direct-comparison inference-and-contextual-alignment

        direct-comparison ::= "#### Direct Comparison for Factual Accuracy:\n" comparison-points
        comparison-points ::= bullet-point+
        bullet-point ::= "  - " comparison-point-detail "\n"
        inference-and-contextual-alignment ::= "#### Inference and Contextual Alignment: " contextual-alignment-detail "\n\n"

        # Final judgment
        final-judgment ::= "### Final Judgment:" "\n" comprehensive-assessment overall-accuracy-determination

        comprehensive-assessment ::= "#### Comprehensive Assessment: " assessment-detail "\n"
        overall-accuracy-determination ::= "#### Overall Accuracy Determination: " accuracy-detail "\n"

        # Terminal symbols
        text-info-detail ::= [^\n]+
        info-type-detail ::= [^\n]+
        answer-detail ::= [^\n]+
        answer-type-detail ::= [^\n]+
        comparison-point-detail ::= [^\n]+
        contextual-alignment-detail ::= [^\n]+
        assessment-detail ::= [^\n]+
        accuracy-detail ::= [^\n]+

        # understand-step ::= "Step " [0-9]?[0-9] ". " "Understand" [^\n]+ "\n"
        """
        )

        answer_constrain_to_text_plan_grammar = LlamaGrammar.from_string(
            r"""
           
        root ::= analyze-step understand-step identify-step plan-revised-step "\n"

        # step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Final" | "Plan" | "Identify" ) [^\n]+ "\n"

        analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze the Text:" [^\n]+ "\n"

        understand-step ::= "Step " [0-9]?[0-9] ". " "Understand the Question:" [^\n]+ "\n"

        identify-step ::= "Step " [0-9]?[0-9] ". " "Identify Flawed Part of the Answer:" [^\n]+ "\n"

        plan-revised-step ::= "Step " [0-9]?[0-9] ". " "Plan Revised Answer:" [^\n]+ "\n"
        """
        )

        answer_relevant_grammar = LlamaGrammar.from_string(
            r"""                     
           
        # Root rule specifying the overall structure of the analysis
        root ::= deep-analysis "\n" comprehensive-understanding "\n" targeted-comparison "\n" identification-of-extraneous-info "\n" final-judgment

        # Deep analysis of the text
        deep-analysis ::= "### Deep Analysis of the Text:" "\n" content-scope-and-detail type-of-information

        content-scope-and-detail ::= "#### Content Scope and Detail: " text-detail "\n"
        type-of-information ::= "#### Type of Information: " info-type "\n"

        # Comprehensive understanding of the answer
        comprehensive-understanding ::= "### Comprehensive Understanding of the Answer:" "\n" key-components-identification depth-of-explanation

        key-components-identification ::= "#### Key Components Identification: " components-detail "\n"
        depth-of-explanation ::= "#### Depth of Explanation: " explanation-detail "\n"

        # Targeted comparison of answer with text
        targeted-comparison ::= "### Targeted Comparison of Answer with Text:" "\n" content-alignment depth-alignment

        content-alignment ::= "#### Content Alignment: " alignment-detail "\n"
        depth-alignment ::= "#### Depth Alignment: " depth-detail "\n"

        # Identification of extraneous information
        identification-of-extraneous-info ::= "### Identification of Extraneous Information:" "\n" spotting-additional-details assessing-impact

        spotting-additional-details ::= "#### Spotting Additional Details: " additional-details "\n"
        assessing-impact ::= "#### Assessing Impact of Additional Information: " impact-assessment "\n"

        # Final judgment on answer relevance
        final-judgment ::= "### Final Judgment on Answer Relevance:" "\n" relevance-assessment explanation-of-judgment

        relevance-assessment ::= "#### Relevance Assessment: " relevance-detail "\n"
        explanation-of-judgment ::= "#### Explanation of Judgment: " judgment-detail "\n"

        # Terminal symbols
        text-detail ::= [^\n]+
        info-type ::= [^\n]+
        components-detail ::= [^\n]+
        explanation-detail ::= [^\n]+
        alignment-detail ::= [^\n]+
        depth-detail ::= [^\n]+
        additional-details ::= [^\n]+
        impact-assessment ::= [^\n]+
        relevance-detail ::= [^\n]+
        judgment-detail ::= [^\n]+

        """
        )

        character_card_grammar = LlamaGrammar.from_string(
            r"""
           
        # Testing making traits come BEFORE dialogue examples, unlike AliChat, so that way it kind of "flows" into dialogue; and also the details are closer to the start and thus more easily remembered.                                     
        #root ::= "Name: " name "\n" "Traits: " traits "\n\nDialogue Examples:" dialogue-examples

        root ::= name "\n" "Traits: " traits "\n\nDialogue Examples:" dialogue-examples

        # Spaces are forbidden in names because during Principles of Chemistry, the script wouldn't stop making the character have the last name Mendeleev!!!
        name ::= [^\n ]+

        traits ::=  trait trait trait trait trait trait trait trait trait trait trait trait trait? trait? trait? trait? trait? trait? trait? trait? # 14 comma-separated traits

        trait ::= [A-Z][a-z ']+ ", " # TODO, it wants hyphens, I can tell because I see it using double spaces for things like Mid  twenties. But idk how to add hyphens to gbnf. Maybe \- ?

        dialogue-examples ::= history personality

        history ::= "\nStranger: \"What's your backstory?\"\n" name ": \"" [^\n]+
        personality ::= "\nStranger: \"What's your personality?\"\n" name ": \"" [^\n]+

        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        character_card_plan_grammar = LlamaGrammar.from_string(
            r"""
           
        # Testing making traits come BEFORE dialogue examples, unlike AliChat, so that way it kind of "flows" into dialogue; and also the details are closer to the start and thus more easily remembered.       





        root ::= [^\n]+ "\n"


        # root ::= consider-step theme-step consistency-step "\n"

        # consider-step ::= "Step 1. " "Consider the provided" [^\n]+ "\n"

        # theme-step ::= "Step 2. " "Given the question, answer, and overall text, a theme for " [^\n]+ "\n"

        # consistency-step ::= "Step 3. " "For this (fictional) character's theme to be what it is, and for them to understand what they do, they would need to live " [^\n]+ # leaving "they must live" relatively open-ended (not "in" or "during") so that this can adapt to even fictional worlds.

        # freeflow-reasoning ::= "Step 4. " "Therefore, a promising character for this question is: " [^\n]+ "\n"



                              
    # root ::= consider-step theme-step step+ "\n"

    # step ::= "Step " [0-9]?[0-9] ". " ( "A Physical Trait" | "One potential detail" | "Another potential detail" | "A potential detail" | "Therefore" | "Note" ) [^\n]+ "\n"

    # consider-step ::= "Step " [0-9]?[0-9] ". " "Consider" [^\n]+ "\n"

    # theme-step ::= "Step " [0-9]?[0-9] ". " "A theme " [^\n]+ "\n"

    """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        # We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
        check_qatuple_context_grammar = LlamaGrammar.from_string(
            r"""
                            
        # Root rule specifying the overall structure of the reasoning and thought process
        root ::= question-validation "\n" answer-validation "\n" critical-evaluation "\n" revised-qatuple

        # Question context validation
        question-validation ::= "### Question Context Validation" "\n" special-term-check-question text-and-author-specificity scope-and-precision

        special-term-check-question ::= "#### Special Term Context Check: Specifically check for use of the terms \"book\", \"text\", \"passage\", and \"excerpt\" without context about which specific thing is being discussed. " question-term-detail "\n"
        text-and-author-specificity ::= "#### Text and Author Specificity: " question-text-author-detail "\n"
        scope-and-precision ::= "#### Scope and Precision: " question-scope-detail "\n"

        # Answer context validation
        answer-validation ::= "### Answer Context Validation:" "\n" special-term-check-answer specificity-and-clarity answer-only-context-issues

        special-term-check-answer ::= "#### Special Term Context Check: Specifically check for use of the terms \"book\", \"text\", \"passage\", and \"excerpt\" without context about which specific thing is being discussed. " answer-term-detail "\n"
        specificity-and-clarity ::= "#### Specificity and Clarity: " answer-specificity-detail "\n"
        answer-only-context-issues ::= "#### Answer-Only Context Issues: " answer-context-issue-detail "\n"

        # Critical evaluation and final judgment
        critical-evaluation ::= "### Critical Evaluation and Final Judgment:" "\n" evaluation final-judgment

        evaluation ::= "#### Evaluation: " evaluation-detail "\n"
        final-judgment ::= "#### Final judgment: " judgment-detail "\n"

        # Optional revised Q&A tuple
        revised-qatuple ::= "### Question Rewording (using text details as reference):" "\n" revised-question-answer

        # Terminal symbols
        question-term-detail ::= [^\n]+
        question-text-author-detail ::= [^\n]+
        question-scope-detail ::= [^\n]+
        answer-term-detail ::= [^\n]+
        answer-specificity-detail ::= [^\n]+
        answer-context-issue-detail ::= [^\n]+
        evaluation-detail ::= [^\n]+
        judgment-detail ::= ("Pass."|"Fail."|"Reword.")
        revised-question-answer ::= "Question: " [^\n]+ "\n" "Answer: " [^\n]+ "\n"

        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        # We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
        check_qatuple_context_reword_grammar = LlamaGrammar.from_string(
            r"""
                            
        # Root rule specifying the overall structure of the reasoning and thought process
        root ::= "Question: " [^\n]+ "\n" "Answer: " [^\n]+ "\n"

        """
        )

        ensure_answer_consistent_grammar = LlamaGrammar.from_string(
            r"""                     
           
        root ::= understand-question-step compare-question-step understand-answer-step compare-step final-step "\n"

        # step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Final" ) [^\n]+ "\n"

        understand-question-step ::= "Step " [0-9]?[0-9] ". " "Understand the provided question:" [^\n]+ "\n"

        compare-question-step ::= "Step " [0-9]?[0-9] ". " "Compare the conversation's question: " [^\n]+ "\n"

        understand-answer-step ::= "Step " [0-9]?[0-9] ". " "Understand the provided answer:" [^\n]+ "\n"

        # compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the " ("first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth") " Part of the Answer with the Text: check if the text " [^\n]+ "\n"

        compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the conversation's answer:" [^\n]+ "\n"

        final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: " ("Inconsistent" | "Consistent") "\n"
        """
        )

        ensure_multiple_answers_consistent_grammar = LlamaGrammar.from_string(
            r"""                     
        
        # Root rule to define the overall structure
        root ::= sequential-matching-section accuracy-check-section conclusion-section

        # Section for sequential matching
        sequential-matching-section ::= "## Sequential Matching of Questions in the Conversation:\n### Sequence and Phrasing of Questions:\n" matching-statement+

        # Section for accuracy check
        accuracy-check-section ::= "## Accuracy Check for Answers in the Conversation:\n### Matching Answers with Provided Content:\n" accuracy-statement+

        # Conclusion section
        conclusion-section ::= "## Conclusion:\n" conclusion-statement+

        # Definitions of different components
        number ::= [1-9]
        matching-statement ::= number ". " [^\n]+ "\n"
        accuracy-statement ::= number ". " [^\n]+ "\n"
        conclusion-statement ::= "  - " [^\n]+ "\n"
        final-judgement ::= "  - Final Judgment:" [^\n]+
        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        # We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
        identify_duplicates_grammar = LlamaGrammar.from_string(
            r"""
           
        # Root rule defining the overall structure of the response
        root ::= normalization-block core-components-block comparative-analysis-block criteria-block conclusion-block

        # Normalization of Questions
        normalization-block ::= "## Normalization of Questions:\n" normalized-question+
        normalized-question ::= "- \"" content "\"\n  - Normalized: " content "\n"

        # Identification of Core Components
        core-components-block ::= "## Identification of Core Components:\n### Subject Matter:\n" subject-matter+ "### Information Sought:\n" information-sought+
        subject-matter ::= "- Question " digit+ ": " content "\n"
        information-sought ::= "- Question " digit+ ": " content "\n"

        # Comparative Analysis Across Questions
        comparative-analysis-block ::= "## Comparative Analysis Across Questions:\n### Direct Comparison:\n" bullet-item+ "### Overlap in Core Components:\n" bullet-item+ "\n"

        # Criteria for Duplication
        criteria-block ::= "## Criteria for Duplication:\n### Exact Information Match:\n" content "### Negation of Minor Differences:\n" content "\n"

        # Conclusion and Labeling
        conclusion-block ::= "## Conclusion and Labeling:\n" content "\n\n## Unique Questions: " unique-questions "\n"
        unique-questions ::= "[" digit+ (", " digit+)* "]"

        # Basic components
        content ::= char+ # A sequence of characters representing content

        digit ::= [0-9] # Digits
        char ::= [^\n] # Any character except newline

        bullet-item ::= "- " content "\n"
        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        judge_paragraph_grammar = LlamaGrammar.from_string(
            r"""                     
           
        root ::= identify-content-step evaluate-relevance-step assess-contexts-and-formats-step assess-possibility-step determine-suitability-step check-contextual-completeness-step final-step "\n"

        identify-content-step ::= "Step " [0-9]?[0-9] ". " "Identify Paragraph Content: " [^\n]+ "\n"

        evaluate-relevance-step ::= "Step " [0-9]?[0-9] ". " "Evaluate Educational Relevance: " [^\n]+ "\n"

        assess-contexts-and-formats-step ::= "Step " [0-9]?[0-9] ". " "Assess Specific Contexts and Formats:" "\n" context-format-bullets

        assess-possibility-step ::= "Step " [0-9]?[0-9] ". " "Assess the Possibility of Formulating Questions: " [^\n]+ "\n"

        determine-suitability-step ::= "Step " [0-9]?[0-9] ". " "Determine Suitability for Educational Purposes: " [^\n]+ "\n"

        check-contextual-completeness-step ::= "Step " [0-9]?[0-9] ". " "Check for Contextual Completeness: " [^\n]+ "\n"

        final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgment: " ("Unsuitable" | "Suitable" | "suitable" | "unsuitable") "\n"

        context-format-bullets ::= bullet-item+
        bullet-item ::= "  - " bullet-item-detail "\n"
        bullet-item-detail ::= [^\n]+
        """
        )

        make_regenerate_answer_plan_grammar = LlamaGrammar.from_string(
            r"""
        root ::= analyze-step understand-step identify-step plan-step

        analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze the Text:" [^\n]+ "\n"

        understand-step ::= "Step " [0-9]?[0-9] ". " "Understand the Question:" [^\n]+ "\n"

        identify-step ::= "Step " [0-9]?[0-9] ". " "Identify the Incorrect Part of the Answer:" [^\n]+ "\n"

        plan-step ::= "Step " [0-9]?[0-9] ". " "Plan a Corrected Answer:" [^\n]+ "\n"
        """
        )

        make_regenerate_question_plan_grammar = LlamaGrammar.from_string(
            r"""
        root ::= analyze-step identify-step generate-step refine-step ensure-step end-of-reasoning

        step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Identify" | "Generate" | "Refine" | "Ensure" ) [^\n]+ "\n"

        analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze the Reason for the Flaw:" [^\n]+ "\n"

        identify-step ::= "Step " [0-9]?[0-9] ". " "Identify Key Concepts in Paragraphs:" [^\n]+ "\n"

        generate-step ::= "Step " [0-9]?[0-9] ". " "Generate a New Question Idea:" [^\n]+ "\n"

        refine-step ::= "Step " [0-9]?[0-9] ". " "Refine the Question:" [^\n]+ "\n"

        ensure-step ::= "Step " [0-9]?[0-9] ". " "Ensure Alignment with Text:" [^\n]+ "\n"

        end-of-reasoning ::= "Step " [0-9]?[0-9] ". " "End" [^\n]+ "\n"
        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        multi_turn_conversation_grammar = LlamaGrammar.from_string(
            r"""

        # The root rule defines the structure of the dialogue
        root ::= [^\n]+ ":" [^\n]+ #"\n" statement+# statement anything+ # Idea: get it started off right, then  let it end how it wants

        """
        )

        proofread_output_grammar = LlamaGrammar.from_string(
            r"""                     
           
        root ::= analyze-step step+ "\n\nBegin Edit: " [^\n]+

        step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Notice" | "Note" | "There is" | "Error" | "I found" | "End" | "There are" ) [^\n]+ "\n"

        analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze" [^\n]+ "\n"
        """
        )

        question_grammar = LlamaGrammar.from_string(
            r"""
        root ::= (question-one answer "\n")

        # Define the question structure with a number followed by content and ending punctuation
        # question ::= number ".) " [^\n]+ [?.!] "\n" # maybe blacklist ?!. along with newlines

        # Define the answer structure
        answer ::= "Answer: " [^\n]+ "\n"

        # Define a number (in this case, limiting to any three-digit number for simplicity)
        number ::= [1-9] [0-9]? [0-9]?

        # Define content as a sequence of characters excluding the word "paragraph" and using not_paragraph to build up the content
        # content ::= (not-paragraph "paragraph")* #not_paragraph


        question-one ::= "1.) " [^\n]+ [?.!] "\n" # maybe blacklist ?!. along with newlines
        # ws ::= [ \t\n]*
        # Define not_paragraph as any sequence of characters that does not contain "paragraph" 
        # and is terminated by a space, punctuation or newline to avoid partial matching of the word.
        # not-paragraph ::= ([^\n\ \.\?!]*(["\.\?! ]+[^p\n\ \.\?!]*)* 
        #     ( "p" ([^\an\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "pa" ([^\br\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "par" ([^\ag\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "para" ([^\bg\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "parag" ([^\rr\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "paragr" ([^\aa\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "paragra" ([^\pp\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "paragraph" [^\np\n\ \.\?!]+))* 
        #     [^\n\ \.\?!paragraph]+ 
        
        

        # Optional whitespace: space, tab, or newlines zero or more times
        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        question_plan_grammar = LlamaGrammar.from_string(
            r"""

        # root ::= reasoning-start
        # At least 3 steps
        root ::= identify-step generate-step brainstorm-step relationships-step if-then-step make-suitable-step

        # no-questions-after-here ::= "\nI will not ask any questions about the following information: " [^\n]+ "."

        identify-step ::= "Step " [0-9]?[0-9] ". " "Identify Key Topics:" [^\n]+ "\n"

        # generate-step ::= "Step " [0-9]?[0-9] ". " "Determine Information-Rich Areas:" [^\n]+ "\n"

        brainstorm-step ::= "Step " [0-9]?[0-9] ". " "Brainstorm and Develop Questions Testing Recall:" [^\n]+ "\n"

        relationships-step ::= "Step " [0-9]?[0-9] ". " "Devise Questions" [^\n]+ "\n"

        if-then-step ::= "Step " [0-9]?[0-9] ". " "Create Questions Investigating" [^\n]+ "\n"

        make-suitable-step ::= "Step " [0-9]?[0-9] ". " "Make a Question that Naturally Complements the Text's Focus:" [^\n]+ "\n"

        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        # We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
        question_relevant_grammar = LlamaGrammar.from_string(
            r"""
                            
        # Root rule specifying the overall structure of the reasoning and thought process
        root ::= in-depth-analysis "\n" detailed-understanding "\n" targeted-comparison "\n" critical-evaluation

        # In-depth analysis of the text
        in-depth-analysis ::= "### In-Depth Analysis of the Text:" "\n" content-and-depth type-of-information

        content-and-depth ::= "#### Content and Depth: " text-description "\n"
        type-of-information ::= "#### Type of Information: " information-description "\n"

        # Detailed understanding of the question
        detailed-understanding ::= "### Detailed Understanding of the Question:" "\n" core-requirement depth-of-detail

        core-requirement ::= "#### Core Requirement: " requirement-description "\n"
        depth-of-detail ::= "#### Depth of Detail: " detail-description "\n"

        # Targeted comparison of the question with the text
        targeted-comparison ::= "### Targeted Comparison of the Question with the Text:" "\n" content-match depth-match

        content-match ::= "#### Content Match: " match-description "\n"
        depth-match ::= "#### Depth Match: " depth-match-description "\n"

        # Critical evaluation and final judgment
        critical-evaluation ::= "### Critical Evaluation and Final Judgment:" "\n" judgment

        judgment ::= [^\n]+

        # Terminal symbols
        text-description ::= [^\n]+
        information-description ::= [^\n]+
        requirement-description ::= [^\n]+
        detail-description ::= [^\n]+
        match-description ::= [^\n]+
        depth-match-description ::= [^\n]+
        relevance ::= "Relevant." | "Irrelevant."


                                                
        # root ::= reasoning from-the-text judgement

        # reasoning ::= "First, I will check whether the question is answerable using the information in the paragraphs. The question asks " [^\n]+ "."
        # from-the-text ::= "\nNow, regardless of what my initial thoughts are, I will try to find some passages from the text that directly answer this question, being mindful that \"How\" is different than \"What\". The text has the following information: " [^\n]+ "."
        # judgement ::= "\nAll this considered, the answer is, compared to the provided text," (relevant|irrelevant) "."
        # relevant ::= "relevant" | "Relevant"
        # irrelevant ::= "irrelevant" | "Irrelevant"

        # final_step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: "

        """
        )

        questions_grammar = LlamaGrammar.from_string(
            r"""
        root ::= (question-one answer "\n") (question-two answer "\n") (question-three answer "\n") (question-four answer)

        # Define the question structure with a number followed by content and ending punctuation
        # question ::= number ".) " [^\n]+ [?.!] "\n" # maybe blacklist ?!. along with newlines

        # Define the answer structure
        answer ::= "Answer: " [^\n]+ "\n"

        # Define a number (in this case, limiting to any three-digit number for simplicity)
        number ::= [1-9] [0-9]? [0-9]?

        # Define content as a sequence of characters excluding the word "paragraph" and using not_paragraph to build up the content
        # content ::= (not-paragraph "paragraph")* #not_paragraph


        question-one ::= "1.) " [^\n]+ [?.!] "\n\n" # maybe blacklist ?!. along with newlines

        question-two ::= "2.) " [^\n]+ [?.!] "\n\n" # maybe blacklist ?!. along with newlines

        question-three ::= "3.) " [^\n]+ [?.!] "\n\n" # maybe blacklist ?!. along with newlines

        question-four ::= "4.) " [^\n]+ [?.!] "\n\n" # maybe blacklist ?!. along with newlines

        # ws ::= [ \t\n]*
        # Define not_paragraph as any sequence of characters that does not contain "paragraph" 
        # and is terminated by a space, punctuation or newline to avoid partial matching of the word.
        # not-paragraph ::= ([^\n\ \.\?!]*(["\.\?! ]+[^p\n\ \.\?!]*)* 
        #     ( "p" ([^\an\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "pa" ([^\br\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "par" ([^\ag\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "para" ([^\bg\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "parag" ([^\rr\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "paragr" ([^\aa\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "paragra" ([^\pp\n\ \.\?!]+ ["\.\?! ]+)* 
        #     | "paragraph" [^\np\n\ \.\?!]+))* 
        #     [^\n\ \.\?!paragraph]+ 
        
        

        # Optional whitespace: space, tab, or newlines zero or more times
        """
        )

        regenerate_answer_constrain_to_text_grammar = LlamaGrammar.from_string(
            r"""
                                                
        root ::= reasoning

        reasoning ::= [^\n]+ "\"\"\""
        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        regenerate_answer_grammar = LlamaGrammar.from_string(
            r"""
                                                
        root ::= reasoning

        reasoning ::= [^\n]+ "."
        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        scenario_grammar = LlamaGrammar.from_string(
            r"""

        root ::= reasoning-start

        reasoning-start ::= [^\n\t]+ "."

        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        # TODO ban the word Stranger here, or use a randomized name in the character card. OR get an LLM to generate a name for the character card.
        scenario_plan_grammar = LlamaGrammar.from_string(
            r"""
                                                     
        root ::= consider-question-step consider-character-step constrain-step setting-step create-step second-message-step "\n"

        consider-question-step ::= "Step " [0-9]?[0-9] ". " "Focus on the question and answer: The question" [^\n]+ "\n"

        consider-character-step ::= "Step " [0-9]?[0-9] ". " "Character Consideration: The primary character is" [^\n]+ "\n"

        constrain-step ::= "Step " [0-9]?[0-9] ". " "Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from" [^\n]+ "\n"

        setting-step ::= "Step " [0-9]?[0-9] ". " "Setting: Given the subject of the question, and the character card, the setting will be" [^\n]+ "\n"

        create-step ::= "Step " [0-9]?[0-9] ". " "Interaction: Given these constraints, the first message (delivered by the secondary character) might" [^\n]+ "\n"

        second-message-step ::= "Step " [0-9]?[0-9] ". In the second message," [^\n]+ "\n"

        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        # TODO ban the word Stranger here, or use a randomized name in the character card. OR get an LLM to generate a name for the character card.
        scenario_plan_many_tuples_grammar = LlamaGrammar.from_string(
            r"""

        root ::= consider-question-step consider-character-step constrain-step setting-step create-step "\n"

        consider-question-step ::= "Step " [0-9]?[0-9] ". " "Focus on the question and answer:" [^\n]+ "\n"

        consider-character-step ::= "Step " [0-9]?[0-9] ". " "Character Consideration:" [^\n]+ "\n"

        constrain-step ::= "Step " [0-9]?[0-9] ". " "Constrain the Scenario: The interaction" [^\n]+ "\n"

        setting-step ::= "Step " [0-9]?[0-9] ". " "Setting: Given the subject of the question, and the character card, the setting will be" [^\n]+ "\n"

        create-step ::= "Step " [0-9]?[0-9] ". " "Interaction: Given these constraints, the first message might" [^\n]+ "\n"

        """
        )

        ### A grammar that forces the model to generate correct character cards (with traits, names, everything)
        single_turn_conversation_grammar = LlamaGrammar.from_string(
            r"""

        # The root rule defines the structure of the dialogue
        root ::= statement "\n\n" response "\n"

        # Statement by Character Name 1
        statement ::= [^\n]+ ":" [^\n]+

        # Response by Character Name 2
        response ::= [^\n]+ ":" [^\n]+

        # Definition of a character name
        character-name ::= word ("-" word)*
        word ::= [A-Za-z]+
        # Limiting to a maximum of six words
        character-name ::= word | word word | word word word | word word word word | word word word word word | word word word word word word

        # Definition of a dialogue line
        dialogue-line ::= [^\n]+


        """
        )

        ### A GGBNF grammar that forces the model to output text in a particular format
        thought_plan_grammar = LlamaGrammar.from_string(
            r"""

        # Root rule defining the overall structure
        root ::= step+ "\n"

        # Step rule with some text (any characters except newline) followed by a period
        step ::= "Step " [0-9]?[0-9] ". " ("Realize" | "Recognize" | "Conclude" | "Recall" | "Remember" | "Formulate" | "Decompose" | "Break down" | "Break" | "Therefore, the answer is" | "The answer is" | "Realise" | "Calculate" | "Understand" | "Note" | "The plan will") [^\n]+ "\n"

        # Potential way forward: change these reasoning steps to use 
        # step ::= "Step " [0-9]?[0-9] ". " ("Realize" | "Recall" | "Remember" | "Formulate" | "Decompose" | "Break down" | "Break" | "Therefore, the answer is" | "The answer is" | "Realise" | "Calculate" | "Understand" | "Note" | "The plan will") [^\n]+ "\n"

        """
        )


NODE_CLASS_MAPPINGS = {
    # Grammars
    "Grammars": Grammars,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grammars": "Import Grammars"
}
