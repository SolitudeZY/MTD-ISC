    var elevideo = document.getElementById("video");
    var elepdf1=document.getElementById("experience")
    var elepdf2=document.getElementById("instruct_book")
    var elepdf3=document.getElementById("model_info")
    var prestr= location.search;//获取当前地址栏中的“查询字符串”值
    var str =prestr.slice(1);//截取？后面的字符串
    var chapter_id={}; //定义一个空的对象进行接收
    var question_id=[];
    var answer_type=[];
    splits(str)
//列表转字典 封装函数 对参数进行处理
function splits(e){
    var par= e.split('=')  //par=['chapter_id','30']
    chapter_id[par[0]] =par[1]
}