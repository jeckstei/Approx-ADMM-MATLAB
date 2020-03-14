function pd = pathDist(path,len)

    dims = size(path);
    last = dims(2);

    target = path(:,last);

    reptarget = repmat(target,1,len);
    diffpath  = path(:,1:len) - reptarget;
    diffpath2 = diffpath.*diffpath;
    sos       = sum(diffpath2,1);
    pd        = sqrt(sos);

end
