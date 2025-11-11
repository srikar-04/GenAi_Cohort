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



const text = `
    In a shimmering forest lived a unicorn named Lumina with a coat as white as snow. She possessed a spiral horn that glowed with a gentle, moonlit light. One day, a dark shadow crept over the valley, threatening to dim all the stars. The woodland creatures, frightened, turned to Lumina for help. With courage in her heart, Lumina used her magic to create a beam of pure light. The beam pierced the darkness, revealing it to be nothing more than a lost, scared cloud. The cloud dissipated, and peace returned to the forest. Lumina taught everyone that even the smallest light can banish the deepest shadow. The animals cheered for their brave unicorn friend, and the forest shone brighter than ever before. From that day on, Lumina's legend grew, a beacon of hope for all.
`

const prompt = `
    summarize the text delimented in triple quotes into single sentence

    '''${text}'''

`

const delimentedResponse = await getCompletions(prompt)

console.log(delimentedResponse)