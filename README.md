#### Applied attention on Sequence to Sequence model for the task of Text Summarizer using Tensorflow's raw_rnn. here, I haven't used Tensorflow's inbuilt seq2seq function. The reason behind is to apply attention mechanism manually.
#### Loss convergence and output can be seen from "final_output.txt"[output of Seq2Seq_model_for_TextSummarizer-600L.py].
#### Though the loss convergence is not great but this gives the idea of how attention works in a Sequence to Sequence model.
##### Task here > Encoder Dataset : News of length 450-600(min to max) tokens & Decoder Dataset : Summarised headlines of max length 56 tokens
###### p.s. : any suggestions would be helpful to improve the model. 
