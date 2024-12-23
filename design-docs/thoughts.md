DAY 1:
The library can be:
1) a playground to relace each part of "pytorch GPU functions" with your own triton kernel
2) a real tool for upgrading "pytorch GPU functions" 
3) an easy and convinient way of adding extra kernels to pytorch

<br/><br/>

DAY 2:
It seems the best practice for replacing Triton kernels with
pytorch is not by mokey-patching, instead, three options are presented:
1) Torch Compile    --> since version 2.0
2) Torch fx         --> since version 1.8
3) Torch jit        --> since version 1.0 <br/>
    a) Torch jit trace <br/>
    b) Torch jit script

The first one uses torch dynamo and is prefered when 
you want to replace your kernel in any model (although the model should be torch dynamo compatible). <br/>
The fx method is more flexible but it is not a general solution because it cannot handle dynamic shapes and control flow.<br/>
The third one is for for deploying torch models in envoiroments other than python.

TODO: also both Gemini and ChatGPT suggested to use fx, Search about torch compile and its relation to Triton
