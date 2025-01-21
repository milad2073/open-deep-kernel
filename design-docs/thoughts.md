DAY 1:<br/>
The library can be:
1) a playground to replace each part of "pytorch GPU functions" with your own triton kernel
2) a real tool for upgrading "pytorch GPU functions" 
3) an easy and convinient way of adding extra kernels to pytorch

<br/><br/>

DAY 2:<br/>
It seems the best practice for replacing Triton kernels with
pytorch is not by monkey-patching, instead, three options are presented:
1) Torch Compile    --> since version 2.0
2) Torch fx         --> since version 1.8
3) Torch jit        --> since version 1.0 <br/>
    a) Torch jit trace <br/>
    b) Torch jit script

The first one uses torch dynamo and is prefered when 
you want to replace your kernel in any model (although the model should be torch dynamo compatible). <br/>
The fx method is more flexible but it is not a general solution because it cannot handle dynamic shapes and control flow.<br/>
The third one is for for deploying torch models in envoiroments other than python. <br/>
Both Gemini and ChatGPT suggested to use fx. <br/><br/>
<br/>
<br/>

DAY 3:<br/>
TODO: Search about torch compile and its relation to Triton. [Done] <br/>

<br/>
<br/>

DAY 4:<br/>
It seems both torch fx and torch compile graphes does not show the fines element as their nodes. For example instead of showing the "matmul" operation, they show the "linear" function. <br/>

We may need to implemenmt a pass that decompose the high level nodes with the low level nodes. <br/>


<br/>
<br/>
DAY 9:<br/>
torch._inductor may give us an opertunity to access low level operators. <br/>


