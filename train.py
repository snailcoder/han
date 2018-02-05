import tensorflow as tf
import numpy as np
import han_model
import data_utils

tf.flags.DEFINE_string("review_path", "review.txt", "The yelp review file.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.flags.DEFINE_float("keep_prob", 0.5,
                      "The probability that each element is kept.")
tf.flags.DEFINE_integer("gru_size", 50,
                        "The number of units int the GRU cell.")
tf.flags.DEFINE_integer("context_size", 100,
                        "The size of word/sentence context vectors.")
tf.flags.DEFINE_integer("min_count", 5,
                        "The word of which frequency is less than "
                        "min_count will be replaced with <UNK>.")
tf.flags.DEFINE_integer("max_doc_len", 15,
                        "The max sentence number contained in single "
                        "document.")
tf.flags.DEFINE_integer("max_sent_len", 100,
                        "The max word number contained in single sentence.")
tf.flags.DEFINE_integer("embedding_size", 200, "The word embedding size.")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size.")
tf.flags.DEFINE_integer("eval_step", 5,
                        "Evaluate model after this many steps")
FLAGS = tf.flags.FLAGS

def main(_):
    data, vocab = data_utils.load_data(FLAGS.review_path)
    unk_vocab = data_utils.replace_UNK(vocab, FLAGS.min_cnt)
    word_idx_map = data_utils.get_word_idx_map(unk_vocab)
    all_text, all_label = zip(*data)
    indexed_docs = data_utils.docs2mat(all_text, FLAGS.max_doc_len,
                                       FLAGS.max_sent_len, word_idx_map)
    folds = data_utils.split_train_test(indexed_docs, all_label)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # Create essential model and operators.
            han =han_model.HanModel(FLAGS.max_doc_len, FLAGS.max_sent_len,
                                    len(unk_vocab), FLAGS.embedding_size,
                                    FLAGS.learning_rate, FLAGS.keep_prob,
                                    FLAGS.gru_size, FLAGS.context_size,
                                    FLAGS.gru_size, FLAGS.context_size,
                                    all_label.shape[1])
            embedding = han.embedding_from_scratch()
            logits = han.inference(embedding)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            loss_op = han.loss(logits)
            train_op = han.training(loss_op, global_step)
            eval_op = han.evaluate(logits)
            # Now feed the data.
            init = tf.global_variables_initializer()
            sess.run(init)
            for train_data, test_data in folds:
                test_X, test_y = zip(*test_data)
                num_batches = int(
                    np.ceil(len(train_data) / FLAGS.batch_size))
                for epoch in range(FLAGS.num_epochs):
                    shuffled_indices = np.random.permutation(len(train_data))
                    shuffled_data = train_data[shuffled_indices]
                    for i in range(num_batches):
                        beg = i * FLAGS.batch_size
                        end = min((i + 1) * FLAGS.batch_size, len(train_data))
                        cur_batch = shuffled_data[beg : end]
                        X_batch, y_batch = zip(*cur_batch)
                        train_feed_dict = {han.input_X: np.array(X_batch),
                                           han.input_y: np.array(y_batch)}
                        _, loss = sess.run([train_op, loss_op],
                                           feed_dict=train_feed_dict)
                    if (epoch % FLAGS.eval_step == 0
                        or epoch == FLAGS.num_epochs - 1):
                        val_feed_dict = {han.input_X: np.array(test_X),
                                         han.input_y: np.array(test_y)}
                        accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                        

if __name__ == "__main__":
    tf.app.run()