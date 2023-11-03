from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
nltk.download('punkt')

summarizer = TextRankSummarizer()


def get_summary(text: str) -> str:
    sum_sent = ''
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    summary = summarizer(parser.document, 2)
    for sentence in summary:
        sum_sent += (str(sentence)) + ' '
    return sum_sent

