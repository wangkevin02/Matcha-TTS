import os
import yaml

def read_prompt(prompt_path):
    with open(prompt_path, 'r',encoding='utf-8') as f:
        return f.read()

config_dir = os.path.join(os.path.dirname(__file__), 'config')
config_path = os.path.join(config_dir, 'config.yaml')
config = yaml.safe_load(open(config_path))
OPENAI_CONFIG = config['openai']
persona_prompt_path = os.path.join(config_dir, 'persona_prompt.txt')
character_prompt_path = os.path.join(config_dir, 'character_prompt.txt')
direct_prompt_path = os.path.join(config_dir, 'direct_description.txt')
merge_prompt_path = os.path.join(config_dir, 'merge_prompt.txt')
scene_prompt_path = os.path.join(config_dir, 'scene_prompt.txt')
persona_prompt = read_prompt(persona_prompt_path)
character_prompt = read_prompt(character_prompt_path)
direct_prompt = read_prompt(direct_prompt_path)
merge_prompt = read_prompt(merge_prompt_path)
scene_prompt = read_prompt(scene_prompt_path)


zh_persona_prompt_path = os.path.join(config_dir, 'zh_persona_prompt.txt')
zh_character_prompt_path = os.path.join(config_dir, 'zh_character_prompt.txt')
zh_merge_prompt_path = os.path.join(config_dir, 'zh_merge_prompt.txt')


zh_persona_prompt = read_prompt(zh_persona_prompt_path)
zh_character_prompt = read_prompt(zh_character_prompt_path)
zh_merge_prompt = read_prompt(zh_merge_prompt_path)

