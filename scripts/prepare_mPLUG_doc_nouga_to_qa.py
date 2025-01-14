import random
import json
from tqdm.auto import tqdm

class QueryPool:
    """
    Class to generate refined prompts for reformatting, enhancing readability, or improving structure of documents or images.
    """

    # Action verbs to initiate the prompt
    level_1 = [
        'Translate', 'Present', 'Reformat', 'Express', 'Format', 'Extract', 'Interpret',
        'Organize', 'Convert', 'Redesign', 'Enhance', 'Refine', 'Clarify', 'Adjust', 'Rephrase'
    ]

    # Modifiers for the target subject
    level_2 = ["the", "the provided", "this", "the original",
               "the initial", "the input", "the current version of"]

    # Target subjects
    level_3 = ["image", "figure", "picture",
               "document", "content", "material"]

    # Purposes for the reformatting or adjustment
    level_4 = [
        "to Markdown format",
        "to markdown format",
        "with Latex equation",
    ]

    def __init__(self, seed=42):
        self.random = random.Random(seed)

    def get_random_prompt(self):
        """
        Generates a refined prompt by selecting one phrase from each level.
        
        Returns:
            str: A formatted prompt string.
        """
        prompt = ' '.join([
            self.random.choice(self.level_1),
            self.random.choice(self.level_2),
            self.random.choice(self.level_3),
            self.random.choice(self.level_4),
        ])

        # Adds a period at the end with 70% probability
        if self.random.random() < 0.7:
            prompt += "."

        return prompt


if __name__ == "__main__":
    data_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_nougat_clean_remove_dumpicated_pattens.jsonl"
    output_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_nougat_clean_remove_dumpicated_pattens_refined_prompts.jsonl"
    # data_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math_clean.jsonl"
    # output_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math_clean_refined_prompts.jsonl"

    # Example usage:
    query_pool = QueryPool()

    with open(data_path, "r") as istream, open(output_path, "w") as ostream:
        for line in tqdm(istream):
            item = json.loads(line)
            image = item["image"] # type: str
            contex = item["src"] # type: str

            que = query_pool.get_random_prompt()

            ostream.write(json.dumps({
                "image": image,
                "messages": [
                    {
                        "role": "user",
                        "content": que,
                    },
                    {
                        "role": "assistant",
                        "content": contex.strip(),
                    }
                ]
            }) + "\n")
