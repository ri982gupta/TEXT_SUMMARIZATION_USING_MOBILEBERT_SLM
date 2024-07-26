import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def summarizer(rawdocs):
    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load('en_core_web_sm')
    
    # Process the raw text
    doc = nlp(rawdocs)

    # Create a list of stopwords
    stopwords = list(STOP_WORDS)

    tokens = [token.text for token in doc]
    # print(doc)

    # Calculate word frequency
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    # Normalize word frequencies
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    # Calculate sentence scores
    sent_tokens = [sent for sent in doc.sents]
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    # Select top sentences
    select_len = int(len(sent_tokens) * 0.3)
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    # Generate final summary
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    
    # Return summary, original document, original text length, and summary text length
    return summary, doc, len(rawdocs.split()), len(summary.split())


# Test the summarizer function
text = """The internet is the world’s most innovative and engaging innovation. It is the most beneficial technology for sharing knowledge from one part of the world to another. After using the internet, it appears that the globe has shrunk in size and that everyone lives close to us. We can know what is happening all around the world in a matter of seconds if we are sitting in one corner of the planet. Today, 90% of work is handled online or with the use of the internet, and that helped us to survive through the pandemic. Whether to study, play games, or watch a movie, we can rely on the internet. If someone is writing about any given topic but is not sure about the same, you need to just click, and you get all the relevant information. Looking into the growing demand and utility of the internet, the government has also provided free internet connections in public areas like railway stations, workplaces, parks, etc. Even though we have become completely dependent on the internet, we must be very careful not to overuse and also take care that children don’t get addicted to it."""

# summary, doc, original_length, summary_length = summarizer(text)
# print("Original Text Length:", original_length)
# print("Summary Text Length:", summary_length)
# print("\nOriginal Text:\n", text)
# print("\nSummary:\n", summary)



# Technology has played a significant role in shaping the world we live in today. From the development of the wheel to the invention of the internet, technology has played a crucial role in human progress. In this essay, I will discuss the impact of technology on society, the benefits and drawbacks of technology, and the future of technology.
# The impact of technology on society has been both positive and negative. On the one hand, technology has greatly improved our lives in many ways. For example, the invention of the internet has made it possible for people to communicate and share information with each other instantly, regardless of their location. This has greatly improved our ability to work, learn, and connect with others. Additionally, technology has also led to the development of new medical treatments and technologies that have saved countless lives.
# On the other hand, technology has also had a negative impact on society. For example, the widespread use of the internet and social media has led to an increase in cyberbullying and online harassment. Additionally, the constant use of technology can also lead to a decrease in face-to-face interaction and social isolation. Furthermore, technology has also led to the loss of jobs in certain industries and an increase in the digital divide between those who have access to technology and those who do not.
# The benefits of technology are many. Technology has led to the development of new and improved products and services, which has improved our standard of living. Technology has also made it easier to communicate and share information with others, which has helped to connect people across the globe. Additionally, technology has also led to the development of new medical treatments and technologies that have saved countless lives.
# The drawbacks of technology are also many. Technology has led to the loss of jobs in certain industries, and an increase in the digital divide between those who have access to technology and those who do not. Additionally, technology has also led to the increase in cyberbullying and online harassment. Furthermore, technology can also lead to a decrease in face-to-face interaction and social isolation.
# The future of technology is uncertain. However, it is likely that technology will continue to play a significant role in shaping the world we live in. As technology continues to advance, it is likely that we will see new and improved products and services that will improve our standard of living. Additionally, technology will continue to connect people across the globe, making it easier to communicate and share information. However, it is also likely that we will continue to see the negative effects of technology on society, such as the loss of jobs and the digital divide.
# Technology has played a significant role in shaping the world we live in today. While technology has brought many benefits, it has also brought many drawbacks. The future of technology is uncertain, but it is likely that technology will continue to play a significant role in shaping the world we live in. It is important for society to be aware of the negative effects of technology and to work towards finding solutions to these problems. Additionally, governments and organizations have a responsibility to ensure that technology is accessible to all and that it is used in a responsible manner.


# The internet is a global network of computers that connects billions of devices around the world. It allows people to access information and communicate with each other from anywhere in the world.
# The internet has revolutionized the way we live, work, and learn. It has made it possible for us to do things that were once impossible, such as working from home, shopping online, and connecting with friends and family all over the world.
# The internet is also a powerful tool for education and research. Students can use the internet to access information from all over the world, and researchers can use it to collaborate on projects and share their findings.
# The internet is a valuable resource that has made our lives easier and more connected. It is a tool that we should use wisely and responsibly.



