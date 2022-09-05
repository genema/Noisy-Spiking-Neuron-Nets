clear;
%h = waitbar(0, 'processing...');
root = './dvs-cifar10';
dirs = dir(root);
for i = 1: length(dirs)
    data = dir([root, '/', dirs(i).name, '/*.aedat']);
    for j = 1: length(data)
        data(j).name
        str = [num2str((j+(i-4)*1000) / 1e+4), '%']
        %waitbar(((i-4)*1000+j-3) / 1e+4, str);
        [root, '/', dirs(i).name, '/', data(j).name]
        res = dat2mat([root, '/', dirs(i).name, '/', data(j).name]);
        save([root, '/', dirs(i).name, '/', sprintf('%d.mat', j)], "res");
    end
end
%close(h)
