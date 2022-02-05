Picsfile = '..\M151023\Site1\train\Pics\';
snum = 49000;
Pics = zeros(50,50,snum);
for i=1:snum
    i
    Pics(:,:,i) = imresize(double(imread([Picsfile int2str(i) '.bmp']))/256,[50,50],'bilinear');
end
%%
valPicsfile = '..\M151023\Site1\val\valPics\';
vnum = 1000;
valPics = zeros(50,50,vnum);
for i=1:vnum
    i
    valPics(:,:,i) = imresize(double(imread([valPicsfile int2str(i) '.bmp']))/256,[50,50],'bilinear');
end
%%
save Pics Pics -v7.3
save valPics valPics -v7.3