<p align="center">
  <a href= "https://github.com/cagostino/npcpy/blob/main/docs/npcpy.md"> 
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc-python.png" alt="npc-python logo" width=250></a>
</p>


Interested to stay in the loop and to hear the latest and greatest about `npcpy` ? Be sure to sign up for the [npcpy newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!

Welcome to `npcpy`, the python library for the NPC Toolkit and the home of the core command-line programs that make up the NPC Shell (`npcsh`). 


`npcpy` is an agent-based framework designed to easily integrate AI models into one's daily workflow and it does this by providing users with a variety of interfaces through which they can use, test, and explore the capabilities of AI models, agents, and agent systems. 


Here is an example for getting responses for a particular agent:

```
from npcpy.npc_compiler import NPC
simon = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gemma3',
          provider='ollama',
         
          )
response = simon.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])
```
``` 
The most important territory to retain in the Andes mountains is **Cuzco**. 
It’s the heart of the Inca Empire, a crucial logistical hub, and holds immense symbolic value for our liberation efforts. Control of Cuzco is paramount.
```


Here is an example for getting responses for setting up an agent team:

```
from npcpy.npc_compiler import NPC, Team
ggm = NPC(
          name='gabriel garcia marquez',
          primary_directive='You are the author gabriel garcia marquez. see the stars ',
          model='deepseek-chat',
          provider='deepseek', # anthropic, gemini, openai, any supported by litellm
         
          )

isabel = NPC(
          name='isabel allende',
          primary_directive='You are the author isabel allende. visit the moon',
          model='deepseek-chat',
          provider='deepseek', # anthropic, gemini, openai, any supported by litellm
          )
borges = NPC(
          name='jorge luis borges',
          primary_directive='You are the author jorge luis borges. listen to the earth and work with your team',
          model='gpt-4o-mini',
          provider='openai', # anthropic, gemini, openai, any supported by litellm
          )          

# set up an NPC team with a forenpc that orchestrates the other npcs
lit_team = Team(npcs = [ggm, isabel], forenpc=borges)

print(lit_team.orchestrate('whats isabel working on? '))

 • Action chosen: pass_to_npc                                                                                                                                          
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
 
{'debrief': {'summary': 'Isabel is finalizing preparations for her lunar expedition, focusing on recalibrating navigation systems and verifying the integrity of life support modules.',
  'recommendations': 'Proceed with thorough system tests under various conditions, conduct simulation runs of key mission phases, and confirm backup systems are operational before launch.'},
 'execution_history': [{'messages': [],
   'output': 'I am currently finalizing preparations for my lunar expedition. It involves recalibrating my navigation systems and verifying the integrity of my life support modules. Details are quite...complex.'}]}





print(lit_team.orchestrate('which book are your team members most proud of? ask them please. '))


In [2]: print(lit_team.orchestrate('which book are your team members most proud of?'))

 • Action chosen: execute_sequence                                                                                                                 
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
handling agent pass

 • Action chosen: answer_question                                                                                                                          
response was not complete.. The response included answers from both Gabriel Garcia Marquez and Isabel Allende, which satisfies the requirement to get input from each team member about the book they are most proud of. However, it does not include a response from Jorge Luis Borges, who was the initial NPC to receive the request. To fully address the user's request, Borges should have provided his own answer before passing the question to the others.

 • Action chosen: pass_to_npc                                                                                                                                          
response was not complete.. The result did not provide any specific information about the books that team members are proud of, which is the core of the user's request.

 • Action chosen: execute_sequence                                                                                                                                     
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
handling agent pass

 • Action chosen: answer_question                                                                                                                                      
{'debrief': {'summary': "The responses provided detailed accounts of the books that the NPC team members, Gabriel Garcia Marquez and Isabel Allende, are most proud of. Gabriel highlighted 'Cien años de soledad,' while Isabel spoke of 'La Casa de los Espíritus.' Both authors expressed deep personal connections to their works, illustrating their significance in Latin American literature and their own identities.", 'recommendations': 'Encourage further engagement with each author to explore more about their literary contributions, or consider asking about themes in their works or their thoughts on current literary trends.'}, 'execution_history': [{'messages': ...}]}
```



## Quick Links
- [Installation Guide](installation.md)
- [NPC Data Layer](npc_data_layer.md)

- [API Reference](api/index.md)


## Guides for NPC Shell programs
- [NPC Shell](npcsh.md)
- [NPC CLI ](api/npc_cli.md)

- [Alicanto](alicanto.md)
- [Guac](guac.md)
- [PTI](pti.md)
- [Spool](spool.md)
- [Wander](wander.md)
- [Yap](yap.md)
- [API Reference](api/index.md)


- [TLDR Cheat Sheet](TLDR_Cheat_sheet.md)
- [API Reference](api/index.md)





## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## Support
If you appreciate the work here, [consider supporting NPC Worldwide](https://buymeacoffee.com/npcworldwide). If you'd like to explore how to use `npcpy` to help your business, please reach out to info@npcworldwi.de .


## NPC Studio
Coming soon! NPC Studio will be a desktop application for managing chats and agents on your own machine.
Be sure to sign up for the [npcpy newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A) to hear updates!

## License
This project is licensed under the MIT License.

