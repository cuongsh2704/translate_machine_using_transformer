# translate_machine_using_transformer

Chương 1: LÝ THUYẾT : BÀI TOÁN DỊCH MÁY
I.	Cấu trúc chung của bài toán dịch máy

 
Ví dụ : ta muốn dịch một câu “ I love teaching “ ra câu “ thôi thích dạy học” 
Cấu trúc chung mô hình sẽ thường như sau :
	Encoder
	Vector context
	Decoder
 
Bộ encoder sẽ vector hóa câu input , vector này được gọi la vector context. Bộ decoder sẽ chuyển vector context này thành output tương ứng
  

II.	Bài toán dịch máy với các mô hình tuần tự ( RNN, LSTM, ..)
 
Phần encoder sử dụng mô hình RNN (nói là mô hình RNN nhưng có thể là các mô hình cải tiến như GRU, LSTM) và context vector được dùng là hidden states ở node cuối cùng. Phần decoder cũng là một mô hình RNN với s0 chính là context vector rồi dần dần sinh ra các từ ở câu dịch
Có thể theo dõi bằng các hình dưới đây:
 
Mô hình encoder
Với input là “I love teaching” qua model ta được một cuỗi vector context
 
Mô hình decoder
Mô hình này sẽ nhận vector context (vectext) làm đầu vào rồi từ từ dịch .
Quy trình có thể miêu tả như sau :
	Vectext + <start> -> Tôi
	Vectext + <start>+Tôi -> thích
	Vectext + <start>+Tôi + thích -> dạy . . . 
Nhược điểm: 
	context vector được lấy từ step cuối của mạng RNN nên việc decode dựa theo vector với các câu dài, lượng thông tin sẽ không lưu trữ đủ trong 1 vector
	 Bên cạnh đó các mạng RNN còn lấy các thông tin của các câu dài không tốt do Vanishing, .
III.	mô hình attention
1.	Mức độ tương quan giữa các từ input và output
Khi dịch máy, có một số từ trong câu cần dịch không có nhiều ảnh hướng với các từ  còn lại trong câu input 
 
Mức độ quan trọng giữa các từ
Ta thấy từ I có trọng số ảnh hướng lớn tới việc dịch từ tôi, hay từ teaching có ảnh hưởng nhiều tới việc dịch từ dạy và từ học. => Do đó khi dịch mỗi từ ta cần chú ý đến các từ ở câu input tiếng anh và đánh trọng số khác nhau cho các từ để dịch chuẩn hơn.
2.	Cơ chế attention
Sử dụng vector context :
 
Gặp vấn đề với việc context vector không thể lựu hết thông tin của một câu
Vậy cách giải quyết bằng attention như sau
 
Khi này context vector sẽ được tính lại như sau
 
Bên cạnh đó, các a_i cũng sẽ được liên kết với nhau như mô hình RNN
Sau đó sẽ dùng context vector từ đầu ra của hidden cuối cùng để bổ sung cho tính toán
VD : Khi dự đoán từ little , context vector được tính tại c1 . khi này các trọng số a3, a4 sẽ được đánh cao hơn thể hiện sự chủ ý hơn giữa 2 từ này
3.	Cơ chế self-attention 
 
Từ Self đứng trước thể hiện việc nó là một attention nhưng chỉ tập sử dụng trên chính nó. Ví dụ như chỉ tự tìm kiếm mức độ liên quan giữa các từ trong câu input chứ không tìm mức độ liên quan giữa các từ output với input
Self Attention cho phép mô hình khi mã hóa một từ có thể sử dụng thông tin của những từ liên quan tới nó. Ví dụ khi từ nó được mã hóa, nó sẽ chú ý vào các từ liên quan như là mặt trời.
Cac thành phần trong attention gồm : Key (K ), Query(Q), value(V)
 
Query lấy thông tin từ tiếp theo cần dịch trong câu OUTPUT gọi là từ Q_word . 2 bộ Key và Value thể hiện mức độ ảnh hưởng của từ thứ i trong câu OUTPUT lên từ Q_word

Dữ liệu sẽ được tính toán theo luồng sau
 
 
 


Có thể tóm tắt thành sơ đồ sau
 
4.	Cơ chế multihead attention
Là phương pháp attention đến nhiều từ trong câu bằng cách sử dụng nhiều cụm self-attention.  
Các ma trận attention sẽ được concat lại, sau đó được nhân với ma trận Wo , ma trận này thể học . cuôi cùng ta được ma trận Z lưu lại thông tin để chuyển vào FeedForward Neural Netword
IV.	Encoding and Decoding trong Transformer model
 

1.	Cấu trúc của Transformer model
 
Các thành phần chính trong một bộ encoder gồm
	Embedding
	Positional
	Multi-head attention
	Normalize
	FFNN(feedforward neral netword)
Trong cấu trúc này thì 2 phần Embedding , multi-head , FFNN đã được giải thích ở phần trước nên không còn xa lạ. Tuy nhiên xuất hiện bộ Positional .
	Về ý nghĩa thì bộ này có tác dụng làm rõ vị trí của một từ trong câu thì với model Transformer,  các trọng số đại diện cho các từ là các ma trận và được train song song trong quá trính cập nhật

2.	Positional: Định vị một từ trong câu
Để xác định vị trí vào cùng wordembedding, ta có thể cộng thêm vector vị trí như [0,1,2,3,...n] . Tuy nhiên như vậy thì số sẽ khá lớn khi độ dài câu lớn, ta có thể chuẩn hóa lại giá trị trong khoảng [0,1] bằng cách chia cho N. Tuy nhiên phương pháp này lại trả kết quả vector hóa một từ liên quan tới độ dài của N. Vì các câu không bằng nhau nên điều này sẽ ảnh hưởng tới quá trình training sau đó
3.	Phương pháp đề xuất sinusoidal position encoding
Về phương pháp sẽ như sau
 
Ta nối thêm một vector vị trí vào sau word embedding, vector này được định nghĩa
 
Trong đó pospos là vị trí của từ trong câu, PE là giá trị phần tử thứ i trong embeddings có độ dài dmodel.
Sở dĩ có điều này là do sự tương đồng giữa biểu diễn bit số và biểu diễn giá trị cosin
 

 
Độ màu sắc sẽ được thay đổi tương ứng với độ thay đổi bit

4.	 Bộ encoder trong transformer
1.	Quá trình encoder
 
Cấu tạo trong bộ encoder
Bao gồm các khối : attention, normalize, các layer Linear

 
Cấu trúc một bộ encoder
Luồng dữ liệu:
	Z = multi-head attention ( X )
	N = Normalize( X + Z )
	F = FFNN( N )
	O = Normalize( N + F )

Output đầu ra của encoder sẽ bằng số từ đầu vào của input
Vd input.shape = (10,256) => output.shape(10,d) (d sẽ phụ thuộc vào lớp FFNN)
5.	Bộ decoder trong transformer
1.	Quá trình decoder
 

Cấu trúc bộ decoder
 
Cấu trúc này có vẻ cũng có các bộ của encoder, tuy nhiên bổ xung thêm :
-	Marked multi-head
Thay vì dùng multi-head, marked multi-head là một bộ che đi các từ trong câu. Cách thức này là vì bộ multi-head attention sẽ chú ý tới toàn bộ các câu trong output, nhưng ta lại chỉ muốn nó dự đoán từng từ trong quá trình dự đoán
 
Chương 2 :TRIỂN KHAI DỊCH MÁY TRÊN MÔ HÌNH TRANSFORMER:

Link code triển khai và bộ dữ liệu 
https://github.com/talonnoxos/translate_machine_using_transformer.git
I.	Chuẩn bị data
1.	Bộ data
Bộ dư liệu việt-nhật với khoảng 2000 câu
 

2.	Tạo từ điển – vocab
Các từ trong câu được tách ra bởi 2 thư viện
  : cho tiếng việt
  : cho tiếng nhật
Danh sách các tử được tách sẽ được đánh số theo tần số xuất hiện và xây dựng nên từ điển bởi thư viện tensorflow.keras.preprocessing.text
 

Và sau đây là kết quả của 2 bộ từ điển
  
3.	Mã hóa và Padding câu
Để thuận lợi cho input của model ta cần padding các câu về cùng kích cỡ
Các câu sau khi mã hóa sẽ padding thêm ký tự { “0” – ‘’ } 
Sử dụng thư viện   tensorflow.keras.preprocessing.sequence để làm việc này
 

Và đây là kết quả một câu sau khi được padding 
 
II.	Tạo bộ positional
Theo như phần lý thuyết đã đề cập, chúng ta cần sử dụng một bộ đánh vị trí cho các câu input .
Triển khai bộ này như sau :
Tạo 2 ma trận sin-cosin dựa theo vị trí 
 
Sau đó cộng các câu mã hóa được embedding tại với ma trận positional trên
 

Đây là kết quả của một ma trận positional

 
III.	Khởi tạo bộ masking
Ma trận masking là là thận chéo giúp che đi các từ trong câu input_target 

Ma trận sẽ có dạng như sau :
 

Với giá trị 0 là giá trị cần dùng, giá trị “ -inf ” là các giá trị cần che
Bên cạnh đó ta cũng cần ma trận để che đi các từ được padding trong câu là các ký tự { 0 – ‘’ } được thêm vào. Các giá trị được che đi sẽ đánh là TRUE, ngươc lại cần dùng là FALSE
 
IV.	Khởi tạo model
Cấu trúc model sẽ như sau:
Gồm các bộ chính :
-	Positional 
-	Embedding
-	Transformer Encoder
-	Transformer Decoder
-	Các layer Linear
Transformer(
  (positional_encoder_vi): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (positional_encoder_ja): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (embedding_vi): Embedding(1634, 256)
  (embedding_ja): Embedding(1996, 256)
  (transformer): Transformer(
    (encoder): TransformerEncoder(
      (layers): ModuleList(
        (0): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (1): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (2): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): TransformerDecoder(
      (layers): ModuleList(
        (0): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
        )
        (1): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
        )
        (2): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
  )
  (out): Sequential(
    (0): Linear(in_features=256, out_features=1634, bias=True)
    (1): Softmax(dim=2)
  )
)
-	Đầu ra của model là xác xuất của mỗi từ trong bộ từ điển

-	Về dữ liệu đầu vào : 
Gồm 3 phần ( đã được mã hóa và padding )
Src : câu tiếng Nhật cần dịch
Với shape 
Tgt_input : câu tiếng Việt mục tiêu
Với shape 

Tgt_ouput là output dự đoán của model. Tuy nhiên sẽ được one-hot tại mỗi mã của từ . Do output của model sẽ dự đoán xác suất của từ đó 
 
Với vi_vocab_size = 1643
 
V.	Kết quả

Kêt quả loss trên 2 tập train-val
 
Đây là kết quả dự đoán một câu trong tệp validation
 
