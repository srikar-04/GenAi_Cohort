import dotenv from 'dotenv'
dotenv.config()
import OpenAI from "openai";

const apiKey = process.env.GEMINI_API_KEY

if(!apiKey) throw new Error('api key not found')

const client = new OpenAI({
    apiKey: apiKey,
    baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
})

async function getCompletions(prompt) {

    const message = [{
        role: "user",
        content: prompt
    }]

    const response = await client.chat.completions.create({
        model: "gemini-2.0-flash",
        messages: message
    })

    return response.choices[0].message.content
}



const text_1 = `
    In a shimmering forest lived a unicorn named Lumina with a coat as white as snow. She possessed a spiral horn that glowed with a gentle, moonlit light. One day, a dark shadow crept over the valley, threatening to dim all the stars. The woodland creatures, frightened, turned to Lumina for help. With courage in her heart, Lumina used her magic to create a beam of pure light. The beam pierced the darkness, revealing it to be nothing more than a lost, scared cloud. The cloud dissipated, and peace returned to the forest. Lumina taught everyone that even the smallest light can banish the deepest shadow. The animals cheered for their brave unicorn friend, and the forest shone brighter than ever before. From that day on, Lumina's legend grew, a beacon of hope for all.
`

const delimentedPrompt = `
    summarize the text delimented in triple quotes into single sentence

    '''${text_1}'''

`

const delimentedResponse = await getCompletions(delimentedPrompt)

// console.log(delimentedResponse)

const text_2 = `
    Making a cup of tea is easy! First, you need to get some \ 
    water boiling. While that's happening, \ 
    grab a cup and put a tea bag in it. Once the water is \ 
    hot enough, just pour it over the tea bag. \ 
    Let it sit for a bit so the tea can steep. After a \ 
    few minutes, take out the tea bag. If you \ 
    like, you can add some sugar or milk to taste. \ 
    And that's it! You've got yourself a delicious \ 
    cup of tea to enjoy.
`

const conditionalPrompt = `
    you will be provided with a text deliminted in triple quotes. 
    If it contains a sequence of instructions, rewrite those instructions in the following format.

    Step - 1 - ....
    Step - 2 - ....
    
    ...
    Step - N - ....

    if the text does not contain sequence of instructions then simply output - "NO SEQUENCE OBSERVED" 

    """${text_2}"""
`

const conditionalResponse = await getCompletions(conditionalPrompt)

// console.log(conditionalResponse)

const structuredPrompt = `
    give me a list of 3 items along with author name and year of release.
    Provide them in json format with the following keys
    id, title, author_name and year_of_release
`

const structuredResponse = await getCompletions(structuredPrompt)

console.log(structuredResponse)
console.log(typeof structuredResponse)