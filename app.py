from flask import Flask, render_template, request, jsonify
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer