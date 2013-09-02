function [ cell_part ] = vecToCellPartition( v, n_part_elem )
%VECTOCELLPARTITION Summary of this function goes here
%   Detailed explanation goes here
    v = v(:);
    lv = length(v);
    n_rows = floor(lv/n_part_elem);
    felem = n_rows*n_part_elem;
    v2 = v(1:felem);
    v_mat = reshape(v2', n_part_elem, n_rows)';
    cell_part = mat2cell(v_mat, ones(1, n_rows));
    if felem ~= lv
        cell_part{end+1} = v(felem+1:end);
    end
end

