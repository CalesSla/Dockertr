FROM python:3.9

RUN pip install numpy && \
    pip install pandas && \
    pip install scikit-learn && \
    pip install joblib

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY test.py ./test.py

# RUN python3 train.py
# RUN python3 test.py

# CMD ["python3", "train.py"]
# CMD ["python3", "test.py"]