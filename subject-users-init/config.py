# tags to consider for user definition
bool_cols = ['creative_writing', 'complexity', 'creativity', 'domain_knowledge', 
 'problem_solving', 'real_world', 'specificity', 'technical_accuracy', 
 'math', 'instruction_following','is_code']

# all tags
# ['creative_writing', 'complexity', 'creativity', 'domain_knowledge', 
#  'problem_solving', 'real_world', 'specificity', 'technical_accuracy', 
#  'math', 'instruction_following','is_code']

# grouping initialization
# if any of tags in group 1 are true, assign to group 1
# if any of tags in group 0 are true, assign to group 0
group_tags = {
    1: ['problem_solving', 'creativity', 'technical_accuracy'],
    0: ['creative_writing'],
}

# MMLU subject grouping
HUMANITIES = {"business_ethics", "human_sexuality", "jurisprudence", "management",
              "moral_disputes", "philosophy", "sociology", "world_religions"}
STEM       = {"abstract_algebra", "college_computer_science", "college_mathematics",
              "college_physics", "formal_logic"}

HUMANITIES_GROUP = 0
STEM_GROUP = 1