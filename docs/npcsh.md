<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/npcsh_sibiji.png" alt="npcsh logo with sibiji the spider" width=400 height=400>
</p>


# npcsh


- `npcsh` is a python-based AI Agent framework designed to integrate Large Language Models (LLMs) and Agents into one's daily workflow by making them available and easily configurable through a command line shell as well as an extensible python library.

- **Smart Interpreter**: `npcsh` leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.

- **Macros**: `npcsh` provides macros to accomplish common tasks with LLMs like voice control (`/yap`), image generation (`/vixynt`), screenshot capture and analysis (`/ots`), one-shot questions (`/sample`), computer use (`/plonk`),  retrieval augmented generation (`/rag`), search (`/search`) and more. Users can also build their own jinxs and call them like macros from the shell.


- **NPC-Driven Interactions**: `npcsh` allows users to take advantage of agents (i.e. NPCs) through a managed system. Users build a directory of NPCs and associated jinxs that can be used to accomplish complex tasks and workflows. NPCs can be tailored to specific tasks and have unique personalities, directives, and jinxs. Users can combine NPCs and jinxs in assembly line like workflows or use them in SQL-style models.

* **Extensible with Python:**  `npcsh`'s python package provides useful functions for interacting with LLMs, including explicit coverage for popular providers like ollama, anthropic, openai, gemini, deepseek, and openai-like providers. Each macro has a corresponding function and these can be used in python scripts. `npcsh`'s functions are purpose-built to simplify NPC interactions but NPCs are not required for them to work if you don't see the need.

* **Simple, Powerful CLI:**  Use the `npc` CLI commands to run `npcsh` macros or commands from one's regular shell. Set up a flask server so you can expose your NPC team for use as a backend service. You can also use the `npc` CLI to run SQL models defined in your project, execute assembly lines, and verify the integrity of your NPC team's interrelations. `npcsh`'s NPCs take advantage of jinja templating to reference other NPCs and jinxs in their properties, and the `npc` CLI can be used to verify these references.

* **Powerful jinx integrations:** `npcsh` has built-in jinxs for users to have agents execute code, analyze data, generate images, search the web, and more. jinxs can be defined in YAML files as part of project-specific `npc_team`s or in the global `~/.npcsh/npc_team` directory or simply in python scripts. Once compiled, the jinxs can be used as macros in the `npc` cli as well as `/{jinx_name}` commands in the `npcsh` shell.



# npcsh usage
In the `npcsh` shell, users can ask LLMs questions, have LLMLs execute commands or use jinxs, or utilize macros that provide additional functionality. When a user does not invoke a specific macro, the shell will automatically decide which jinx to use based on the user's input. Here are some examples of things one might ask the npcsh shell.

Here are some examples of how you can use npcsh
```npcsh
npcsh>Who was Simon Bolivar?

Simón Bolívar, often referred to as "El Libertador," was a Venezuelan military and political leader who played a key role in the Latin American wars of independence against Spanish
colonial rule in the early 19th century. He was born on July 24, 1783, in Caracas, Venezuela, into a wealthy Creole family.
Bolívar's vision extended beyond merely liberating territories; he aspired to unify the newly independent nations of South America into a single federation, which he called "Gran
Colombia," consisting of present-day Colombia, Venezuela, Ecuador, and Panama. He was known for his exceptional military strategies and leadership, which led to successful campaigns in
various regions, including the battles of Boyacá, Carabobo, and Ayacucho.
He faced numerous challenges, including political fragmentation, regional rivalries, and personal betrayals. Despite his initial successes, the unity he sought for Latin America proved
difficult to achieve, and Gran Colombia eventually disintegrated in the early 1830s.
Bolívar's influence and legacy extend far beyond his lifetime. He is celebrated in various countries across Latin America as a national hero and symbol of independence. His thoughts on
liberty, governance, and social issues continue to inspire movements for democracy and social justice in the region. Simón Bolívar died on December 17, 1830, but remains a pivotal figure
in the history of Latin America.
```


```npcsh
npcsh> What is the capital of France?
The capital of France is Paris. It is the largest city in the country and is known for its rich history, art, culture, and architecture, including famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
```

```npcsh
npcsh> can you tell me a joke about my favorite city?

Additional input needed: The user did not specify their favorite city, which is necessary to generate a relevant joke.
Please tell me your favorite city so I can share a joke about it!: boston

Sure! Here's a joke about Boston:
Why do Bostonians like to play hide and seek?
Because good luck hiding when everyone yells, "Wicked awesome, ya gotta be here!"
```

```npcsh
npcsh> What's the weather in Tokyo?

handle_jinx_call invoked with jinx_name: generic_search_jinx

The weather in Tokyo, Japan, is expected to be mixed with sun and clouds. Here are some details from the recent forecasts:

 • Highs: Around 53°F to 58°F with a few variations depending on the day.

 • Lows: Approximately 35°F to 40°F.

 • Winds: Generally from the northwest at 5 to 10 mph.

 • Condition: Mainly sunny, but there may be periods of clouds and some overcast conditions throughout the week.
For more detailed information, you can refer to sources like The Weather Channel or AccuWeather.
/home/caug/npcww/npcsh:npcsh>

```
In the below example, the code that was open was the screen capture analysis jinx itself.
```npcsh
npcsh>  Can you explain what the code does in the currently open VS code window?

handle_jinx_call invoked with jinx_name: screen_capture_analysis_jinx

Screenshot saved as screenshot_20241223_225815.png

The code in the visible section of your VS Code window appears to be a script for capturing and analyzing screenshots. Here's a breakdown of what the code does:

 1 Import Necessary Libraries: It imports required libraries like system, datetime, and pyautogui, which are essential for capturing screenshots and handling date-time operations.

 2 Capture the Screen: The code captures the current screen using pyautogui.screenshot(), taking a screenshot of the entire screen.

 3 File Management: It generates a unique filename for the screenshot using the current date and time. This ensures that each screenshot file is distinct.

 4 Save the Screenshot: The captured screenshot is saved to a file with the generated filename.

 5 Image Analysis: It likely imports a functionality (analyze_image_base) to analyze the screenshot for specific content or information.

 6 Run the Analysis: While not explicitly shown, it appears to call this analysis function to process the saved screenshot.
In summary, this code automates the process of capturing a screenshot, saving it with a unique filename, and analyzing that image for specific content or patterns.


```

```npcsh
npcsh>What is the best way to implement a linked list in Python?

The best way to implement a linked list in Python is to define a Node class for the individual elements and a LinkedList class to manage the nodes. Here's a basic implementation:
class Node:
    def __init__(self, data):
        self.data = data  # Store the data
        self.next = None  # Initialize the next node as None

class LinkedList:
    def __init__(self):
        self.head = None  # Initialize the head of the list

    def insert(self, data):
        """Inserts a new node at the end of the linked list."""
        new_node = Node(data)
        if not self.head:  # If list is empty, set new node as head
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:  # Traverse to the last node
            last_node = last_node.next
        last_node.next = new_node  # Link the new node

    def display(self):
        """Prints the linked list."""
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Example usage:
llist = LinkedList()
llist.insert(1)
llist.insert(2)
llist.insert(3)
llist.display()  # Output: 1 -> 2 -> 3 -> None
                                                                                            Explanation:

 1 Node Class: Represents each node in the linked list. It has two attributes: data (to store the value) and next (to point to the next node).

 2 LinkedList Class: Manages the linked list. It has a head attribute pointing to the first node.

 3 Insert Method: Adds a new node to the end of the list.

 4 Display Method: Prints the elements of the list in order.
This implementation is simple and can be extended with more methods (e.g., deletion, searching, etc.) as needed.

```

```npcsh
npcsh> In which of the files in the current directory is "Simon Bolivar" mentioned?

The user's query was about "Simon Bolivar." Based on the relevant context from the codebase, I found specific mentions and relevant information regarding Simon Bolívar in the documentation provided
in the README.md file.
From the relevant context, specifically in the section of README.md, it shared a code example that demonstrates how to load an NPC (Non-Player Character) named "Simon Bolivar" using SQLite and
answer a question about project updates. This code snippet indicates that "Simon Bolivar" is represented in the NPC system, allowing for interaction through questions to retrieve information.
Additionally, there is another passage that directly answers the query. It reads:

▌ "Simón Bolívar, often referred to as 'El Libertador,' was a Venezuelan military and political leader who played a key role in the Latin American wars of independence against Spanish colonial
▌ rule in the early 19th century."
This information highlights Bolívar's significant historical role and achievements, answering the query regarding who Simon Bolivar was, thus making it clear that he was a prominent leader in Latin
America's fight for independence.
To summarize, I arrived at this answer by examining the provided context in the README.md file that includes both the coding example that initializes an NPC named "Simon Bolivar" and a historical
overview of his contributions, which directly addresses the user's inquiry.


```


```npcsh
npcsh>what process is using the most ram on my computer right now?

LLM suggests the following bash command: ps aux --sort=-%mem | head -n 10
Running command: ps aux --sort=-%mem | head -n 10
Command executed with output: USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
caug     1040046  1.1  2.7 1226193464 1598980 ?  SNl  Dec22  23:30 /opt/vivaldi/vivaldi-bin --type=renderer --crashpad-handler-pid=4529 --enable-crash-reporter=,stable --change-stack-guard-on-fork=
caug      683148  0.0  1.9 2047464 1137232 ?     SNl  Dec13  10:11 io.elementary.appcenter
ollama      1884  0.0  1.8 3683920 1051020 ?     SNsl Dec02   0:30 /usr/local/bin/ollama serve
caug     1191836  5.3  1.6 7346052 964988 pts/17 Sl+  22:57   0:22 /home/caug/.pyenv/versions/npcsh/bin/python /home/caug/.pyenv/versions/3.11.0/envs/npcsh/bin/npcsh
caug     1145749  0.0  1.5 7068128 892500 pts/13 Sl+  Dec22   0:15 /home/caug/.pyenv/versions/3.11.0/bin/python3.11 /home/caug/.pyenv/versions/3.11.0/bin/npcsh
caug      493912  0.0  1.5 7050968 890012 pts/11 Sl+  Dec10   0:14 /home/caug/.pyenv/versions/npcsh/bin/python /home/caug/.pyenv/versions/npcsh/bin/npcsh
caug        4522  0.9  1.5 34391236 874756 ?     SNLl Dec02 295:06 /opt/vivaldi/vivaldi-bin
caug        3995  0.7  1.1 6051812 683916 ?      R<sl Dec02 221:19 /usr/bin/gnome-shell
caug      501721  0.3  1.0 1212584660 626248 ?   SNl  Dec10  60:22 /opt/vivaldi/vivaldi-bin --type=renderer --crashpad-handler-pid=4529 --enable-crash-reporter=,stable --change-stack-guard-on-fork=

To determine which processes are using the most RAM on your computer, I executed the command ps aux --sort=-%mem | head -n 10.
This command lists all running processes (ps aux), sorts them by memory usage in descending order (--sort=-%mem), and then retrieves the top 10 entries (head -n 10).
The output shows the processes currently consuming the most memory. In the results, the process using the most RAM is Vivaldi's renderer, followed by other applications such as the Elementary App Center, Ollama server, and several instances of Python related to your NPC shell.
```

```npcsh
npcsh> can you generate an image of a calico cat and an orange tabby cat in the style of a japanese woodblock print where the two cats are engaged in a samurai battle but instead of samurai swords they are holding cat toys that are like wands with felt birds on the ends of them on strings

handle_jinx_call invoked with jinx_name: image_generation_jinx
```
![catfight](test_data/catfight.PNG)


In addition to its ability to execute one-off tasks or jinx calls, npcsh also has
the ability to generate jinx call sequences when asked:
```npcsh

npcsh> can you use a jinx sequence to find the gdp of russia in 2024 and then to use that information to generate an image?

handle_jinx_call invoked with jinx_name: generic_search_jinx
Jinx found: generic_search_jinx
handle_jinx_call invoked with jinx_name: image_generation_jinx
Jinx found: image_generation_jinx

The Gross Domestic Product (GDP) of Russia is estimated to be $8.311 trillion by the end of 2024, according to World Economics.
This figure is significantly larger than the official estimate of $5.804 trillion published by the World Bank for the end of 2023.

It seems that you've generated an image. If you have any questions or need assistance related to this image, please let me know how I can help!
```
and then the associated image :
![gdp](test_data/r8ss9a.PNG)







## Piping outputs
An important facet that makes `npcsh` so powerful is the ability to pipe outputs from one jinx call to another. This allows for the chaining of commands and the creation of complex workflows. For example, you can use the output of a search to generate an image, or you can use the output of an image analysis to generate a report. Here is an example of how this might look in practice:
```npcsh
npcsh> what is the gdp of russia in 2024? | /vixynt 'generate an image that contains {0}'
```
## Executing Bash Commands
You can execute bash commands directly within npcsh. The LLM can also generate and execute bash commands based on your natural language requests.
For example:
```npcsh
npcsh> ls -l

npcsh> cp file1.txt file2.txt
npcsh> mv file1.txt file2.txt
npcsh> mkdir new_directory
npcsh> git status
npcsh> vim file.txt

```


To exit the shell:
```npcsh
npcsh> /exit
```

Otherwise, here are some more detailed examples of macros that can be used in npcsh:
## Conjure (under construction)
Use the `/conjure` macro to generate an NPC, a NPC jinx, an assembly line, a job, or an SQL model

```bash
npc conjure -n name -t 'templates'
```


## Data Interaction and analysis (under construction)


## Debate (under construction)
Use the `/debate` macro to have two or more NPCs debate a topic, problem, or question.

For example:
```npcsh
npcsh> /debate Should humans colonize Mars? npcs = ['sibiji', 'mark', 'ted']
```




## Over-the-shoulder: Screenshots and image analysis

Use the /ots macro to take a screenshot and write a prompt for an LLM to answer about the screenshot.
```npcsh
npcsh> /ots

Screenshot saved to: /home/caug/.npcsh/screenshots/screenshot_1735015011.png

Enter a prompt for the LLM about this image (or press Enter to skip): describe whats in this image

The image displays a source control graph, likely from a version control system like Git. It features a series of commits represented by colored dots connected by lines, illustrating the project's development history. Each commit message provides a brief description of the changes made, including tasks like fixing issues, merging pull requests, updating README files, and adjusting code or documentation. Notably, several commits mention specific users, particularly "Chris Agostino," indicating collaboration and contributions to the project. The graph visually represents the branching and merging of code changes.
```

In bash:
```bash
npc ots
```



Alternatively, pass an existing image in like :
```npcsh
npcsh> /ots test_data/catfight.PNG
Enter a prompt for the LLM about this image (or press Enter to skip): whats in this ?

The image features two cats, one calico and one orange tabby, playing with traditional Japanese-style toys. They are each holding sticks attached to colorful pom-pom balls, which resemble birds. The background includes stylized waves and a red sun, accentuating a vibrant, artistic style reminiscent of classic Japanese art. The playful interaction between the cats evokes a lively, whimsical scene.
```

```bash
npc ots -f test_data/catfight.PNG
```


## Plan : Schedule tasks to be run at regular intervals (under construction)
Use the /plan macro to schedule tasks to be run at regular intervals.
```npcsh
npcsh> /plan run a rag search for 'moonbeam' on the files in the current directory every 5 minutes
```

```npcsh
npcsh> /plan record the cpu usage every 5 minutes
```

```npcsh
npcsh> /plan record the apps that are using the most ram every 5 minutes
```




```bash
npc plan -f 30m -t 'task'
```

Plan will use platform-specific scheduling jinxs. In particular, it uses crontab on Linux and launchd on macOS and Schedule Tasks on Windows.

Implementations have been provided for Mac and Windows but only has been tested as of 3/23/2025 on Linux.



## Plonk : Computer Control
Use the /plonk macro to allow the LLM to control your computer.
```npcsh
npcsh> /plonk go to a web browser and  go to wikipedia and find out information about simon bolivar
```

```bash
npc plonk 'use a web browser to find out information about simon boliver'
```

## RAG

Use the /rag macro to perform a local rag search.
If you pass a `-f` flag with a filename or list of filenames (e.g. *.py) then it will embed the documents and perform the cosine similarity scoring.

```npcsh
npcsh> /rag -f *.py  what is the best way to implement a linked list in Python?
```
```
/rag -f test_data/yuan2004.pdf summarize this paper
```

```
/rag -f test_data/books.csv analyze this dataset
```

Alternatively , if you want to perform rag on your past conversations, you can do so like this:
```npcsh
npcsh> /rag  what is the best way to implement a linked list in Python?
```
and it will automatically look through the recorded conversations in the ~/npcsh_history.db


In bash:
```bash
npc rag -f *.py
```

## Rehash

Use the /rehash macro to re-send the last message to the LLM.
```npcsh
npcsh> /rehash
```

## Sample
Send a one-shot question to the LLM.
```npcsh
npcsh> /sample What is the capital of France?
```

Bash:
```bash
npc sample 'thing' -m model -p provider

```


## Search
Search can be accomplished through the `/search` macro. You can specify the provider as being "perplexity" or "duckduckgo". For the former,
you must set a perplexity api key as an environment variable as described above. The default provider is duckduckgo.

NOTE: while google is an available search engine, they recently implemented changes (early 2025) that make the python google search package no longer as reliable.
Duckduckgo's search toool also givies rate limit errors often, so until a more robust
solution is implemented for it, Perplexity's will be the most reliable.




```npcsh
npcsh!> /search -p duckduckgo  who is the current us president


President Donald J. Trump entered office on January 20, 2025. News, issues, and photos of the President Footer Disclaimer This is the official website of the U.S. Mission to the United Nations. External links to other Internet sites should not be construed as an endorsement of the views or privacy policies contained therein.

Citation: https://usun.usmission.gov/our-leaders/the-president-of-the-united-states/
45th & 47th President of the United States After a landslide election victory in 2024, President Donald J. Trump is returning to the White House to build upon his previous successes and use his mandate to reject the extremist policies of the radical left while providing tangible quality of life improvements for the American people. Vice President of the United States In 2024, President Donald J. Trump extended JD the incredible honor of asking him to serve as the Vice-Presidential Nominee for th...
Citation: https://www.whitehouse.gov/administration/
Citation: https://www.instagram.com/potus/?hl=en
The president of the United States (POTUS)[B] is the head of state and head of government of the United States. the president directs the executive branch of the federal government and is the commander-in-chief of the United States Armed Forces. The power of the presidency has grown substantially[12] since the first president, George Washington, took office in 1789.[6] While presidential power has ebbed and flowed over time, the presidency has played an increasingly significant role in American ...
Citation: https://en.wikipedia.org/wiki/President_of_the_United_States
Citation Links: https://usun.usmission.gov/our-leaders/the-president-of-the-united-states/
https://www.whitehouse.gov/administration/
https://www.instagram.com/potus/?hl=en
https://en.wikipedia.org/wiki/President_of_the_United_States
```


```npcsh
npcsh> /search -p perplexity who is the current us president
The current President of the United States is Donald Trump, who assumed office on January 20, 2025, for his second non-consecutive term as the 47th president[1].

Citation Links: ['https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States',
'https://en.wikipedia.org/wiki/Joe_Biden',
'https://www.britannica.com/topic/Presidents-of-the-United-States-1846696',
'https://news.gallup.com/poll/329384/presidential-approval-ratings-joe-biden.aspx',
'https://www.usa.gov/presidents']
```

Bash:

```bash
(npcsh) caug@pop-os:~/npcww/npcsh$ npc search 'simon bolivar' -sp perplexity
Loaded .env file from /home/caug/npcww/npcsh
urls ['https://en.wikipedia.org/wiki/Sim%C3%B3n_Bol%C3%ADvar', 'https://www.britannica.com/biography/Simon-Bolivar', 'https://en.wikipedia.org/wiki/File:Sim%C3%B3n_Bol%C3%ADvar_2.jpg', 'https://www.historytoday.com/archive/simon-bolivar-and-spanish-revolutions', 'https://kids.britannica.com/kids/article/Sim%C3%B3n-Bol%C3%ADvar/352872']
openai
- Simón José Antonio de la Santísima Trinidad Bolívar Palacios Ponte y Blanco[c] (24 July 1783 – 17 December 1830) was a Venezuelan statesman and military officer who led what are currently the countries of Colombia, Venezuela, Ecuador, Peru, Panama, and Bolivia to independence from the Spanish Empire. He is known colloquially as El Libertador, or the Liberator of America. Simón Bolívar was born in Caracas in the Captaincy General of Venezuela into a wealthy family of American-born Spaniards (crio...
 Citation: https://en.wikipedia.org/wiki/Sim%C3%B3n_Bol%C3%ADvar



Our editors will review what you’ve submitted and determine whether to revise the article. Simón Bolívar was a Venezuelan soldier and statesman who played a central role in the South American independence movement. Bolívar served as president of Gran Colombia (1819–30) and as dictator of Peru (1823–26). The country of Bolivia is named for him. Simón Bolívar was born on July 24, 1783, in Caracas, Venezuela. Neither Bolívar’s aristocrat father nor his mother lived to see his 10th birthday. Bolívar...
 Citation: https://www.britannica.com/biography/Simon-Bolivar



Original file (1,525 × 1,990 pixels, file size: 3.02 MB, MIME type: image/jpeg) Derivative works of this file: Simón Bolívar 5.jpg This work is in the public domain in its country of origin and other countries and areas where the copyright term is the author's life plus 100 years or fewer. This work is in the public domain in the United States because it was published (or registered with the U.S. Copyright Office) before January 1, 1930. https://creativecommons.org/publicdomain/mark/1.0/PDMCreat...
 Citation: https://en.wikipedia.org/wiki/File:Sim%C3%B3n_Bol%C3%ADvar_2.jpg



SubscriptionOffers Give a Gift Subscribe A map of Gran Colombia showing the 12 departments created in 1824 and territories disputed with neighboring countries. What role did Simon Bolivar play in the history of Latin America's independence from Spain? Simon Bolivar lived a short but comprehensive life. History records his extraordinary versatility. He was a revolutionary who freed six countries, an intellectual who argued the problems of national liberation, a general who fought a war of unremit...
 Citation: https://www.historytoday.com/archive/simon-bolivar-and-spanish-revolutions



Known as the Liberator, Simón Bolívar led revolutions against Spanish rule in South America. The countries of Venezuela, Colombia, Ecuador, Panama, Peru, and Bolivia all owe their independence largely to him. Bolívar was born on July 24, 1783, in Caracas, New Granada (now in Venezuela). After studying in Europe, he returned to South America and began to fight Spanish rule. Between 1810 and 1814 Venezuela made two failed tries to break free from Spain. After the second defeat, Bolívar fled to Jam...
 Citation: https://kids.britannica.com/kids/article/Sim%C3%B3n-Bol%C3%ADvar/352872



- https://en.wikipedia.org/wiki/Sim%C3%B3n_Bol%C3%ADvar

https://www.britannica.com/biography/Simon-Bolivar

https://en.wikipedia.org/wiki/File:Sim%C3%B3n_Bol%C3%ADvar_2.jpg

https://www.historytoday.com/archive/simon-bolivar-and-spanish-revolutions

https://kids.britannica.com/kids/article/Sim%C3%B3n-Bol%C3%ADvar/352872
```

```bash
npc search 'snipers on the roof indiana university' -sp duckduckgo
```


## Set: Changing defaults from within npcsh
Users can change the default model and provider from within npcsh by using the following commands:
```npcsh
npcsh> /set model ollama
npcsh> /set provider llama3.2
```


## Sleep : a method for creating and updating a knowledge graph (under construction)

Use the `/sleep` macro to create or update a knowledge graph. A knowledge graph is a structured representation of facts about you as a user that the NPCs can determine based on the conversations you have had with it.
```npcsh
npcsh> /sleep
```

## breathe: a method for condensing context on a regular cadence (# messages, len(context), etc) (under construction)
-every 10 messages/7500 characters, condense the conversation into lessons learned. write the lessons learned down by the np
for the day, then the npc will see the lessons they have learned that day in that folder as part of the context.



## Spool
Spool mode allows one to enter into a conversation with a specific LLM or a specific NPC.
This is used for having distinct interactions from those in the base shell and these will be separately contained.


Start the spool mode:
```npcsh
npcsh> /spool
```
Start the spool mode with a specific npc

```npcsh
npcsh> /spool npc=foreman
```

Start the spool mode with specific files in context that will be referenced through RAG searches when relevant.

```npcsh
npcsh> /spool --files=[*.py,*.md] 
```
/spool --files test_data/yuan2004.pdf

Have a conversation and switch between text and voice mode by invoking `/whisper` mode from within spool mode.
```npcsh
spool> what can you tell me about green bull from one piece?

Green Bull, also known as Ryokugyu, is a character from the popular anime and manga series One Piece. He is one of the Marine Admirals and was introduced during the Wano Country arc. Here are some key points about Green
Bull:
 1 Real Name: His real name is Aramaki, but he is primarily known by his title, Green Bull (Ryokugyu in Japanese).
 2 Appearance: Green Bull has a distinct look characterized by a green military uniform, a long cape, and a
    noticeable plant-based theme, which aligns with his powers.
 3 Devil Fruit: Green Bull is a user of the Mori Mori no Mi (Forest Forest Fruit), a Logia-type Devil Fruit that
    allows him to generate and manipulate plant matter. This ability can be seen in his combat style, which utilizes
    plants to entrap and attack opponents.
 4 Role as an Admiral: As an Admiral, Green Bull holds a high-ranking position in the Marines and is
    known for his strength and authority. He is fiercely loyal to the Marine organization and its goals.
 5 Personality: Green Bull displays a laid-back attitude and often appears more relaxed compared to other Admirals.
    His personality can be somewhat ambiguous, which adds depth to his character.
 6 Involvement in the Story: Green Bull's introduction brings anticipation regarding his potential confrontations with
    key characters in the series. His role in major arcs like Wano and his interactions with the Straw Hat Pirates and
    other significant figures are crucial for the unfolding storyline.
 Overall, Green Bull is an intriguing character with unique abilities and a significant role within the One Piece universe.

spool> /whisper

Calibrating silence level. Please remain quiet...
Silence threshold set to: 5679.193548387097
Listening... (speak now)
...............
Max duration reached.

Processing...
You said: What do you think his greatest weakness is?


Based on what we know about Green Bull from the One Piece series, I can identify a few potential weaknesses:

1. **Extreme Laziness**:
- Green Bull is known for being incredibly lazy, often refusing to move or exert himself unnecessarily.
- This laziness could be exploited by opponents who understand how to manipulate or provoke him into action.

2. **Dependency on External Nutrition**:
- His Devil Fruit power allows him to absorb nutrients from the environment, which suggests he might become weakened in environments with limited plant life or nutrients.
- In extremely barren or non-vegetative settings, his regenerative and plant-based abilities might be significantly reduced.

3. **Pride and Marine Ideology**:
- Like many Marines, he likely has a rigid sense of justice that could be used against him strategically.
- His commitment to Marine principles might make him predictable in certain confrontational scenarios.

4. **Potential Overconfidence**:
- As an Admiral, he might underestimate opponents, especially pirates, due to his high-ranking status.
- His laid-back nature might lead him to not take threats seriously until it's too late.

The most pronounced weakness seems to be his extreme laziness, which could potentially be exploited tactically by skilled opponents who understand how to force him into action or create scenarios that challenge his passive nature.

Here are some additional details about Green Bull (Aramaki) in "One Piece":

 1 Devil Fruit Name:

 • Green Bull's Devil Fruit has been confirmed to be the Mori Mori no Mi, which is a mythical Zoan-type fruit. This allows him to utilize various forms of plant life in combat and has regenerative
   capabilities.

 2 Combat Abilities:

 • His ability to control vegetation gives him significant advantages in battle. He can generate plants to attack or defend and possibly use them for tactical advantages, such as creating barriers
   or entangling enemies.

 3 Reputation:

 • As an Admiral, Green Bull holds a position of great power and authority within the Marines. His reputation is likely formidable among pirates and other opponents.

 4 Interactions with Other Characters:

 • Green Bull's character dynamics with others in the series, particularly with fellow Marines and pirates, can provide insight into his personality and values. His interactions during missions or
   discussions about justice will reveal more about his character.

 5 Appearance and Style:

 • He has a rather unique aesthetic, characterized by his green clothing that symbolizes his connection to nature. His overall appearance contributes to his identity as a plant-based fighter.

 6 Backstory:

 • As of now, specific details about his backstory and how he rose to the rank of Admiral are not extensively explored in the series. This leaves room for further character development and
   background to be unveiled in future chapters.

 7 Ambiguous Personality:

 • While his laziness and laid-back demeanor are evident, it is possible that there are deeper layers to his character that might be revealed through his actions and motivations within the
   overarching narrative of "One Piece."

 8 Role in the Marine Organization:

 • His position as Admiral places him in direct opposition to the main pirate characters, particularly the Straw Hat crew, making him a significant figure in the ongoing conflict between pirates
   and the Marines.
As the story continues to develop, Green Bull's character may evolve and reveal more complexities, weaknesses, and relationships within the world of "One Piece."
```


Start the spool with a specific llm model:
```npcsh
#note this is not yet implemented
npcsh> /spool model=llama3.3
```

```bash
npc spool -n npc.npc
```

## Trigger
Use the /trigger macro to execute specific actionss based on certain conditions.

```npcsh
npcsh> /trigger watch for new PDF downloads in the ~/Downloads directory and move them
to the ~/Documents/PDFs directory . Ensure that the directory exists or create it if it does not.
```

On Linux, trigger makes use of inotify-tools to watch for file system events. On macOS, it uses fswatch, and on Windows, it uses Watch-Command.





## Vixynt: Image Generation
Image generation can be done with the /vixynt macro.

Use /vixynt like so where you can also specify the model to use with an @ reference. This @ reference will override the default model in ~/.npcshrc.

```npcsh
npcsh> /vixynt A futuristic cityscape @dall-e-3
```
![futuristic cityscape](test_data/futuristic_cityscape.PNG)

```npcsh
npcsh> /vixynt A peaceful landscape @runwayml/stable-diffusion-v1-5
```
![peaceful landscape](test_data/peaceful_landscape_stable_diff.png)


Similarly, use vixynt with the NPC CLI from a regular shell:
```bash
$ npc --model 'dall-e-2' --provider 'openai' vixynt 'whats a french man to do in the southern bayeaux'
```




## Whisper: Voice Control
Enter into a voice-controlled mode to interact with the LLM. This mode can executet commands and use jinxs just like the basic npcsh shell.
```npcsh
npcsh> /whisper
```




## Compilation and NPC Interaction
Compile a specified NPC profile. This will make it available for use in npcsh interactions.
```npcsh
npcsh> /compile <npc_file>
```
You can also use `/com` as an alias for `/compile`. If no NPC file is specified, all NPCs in the npc_team directory will be compiled.

Begin a conversations with a specified NPC by referencing their name
```npcsh
npcsh> /<npc_name>:
```


