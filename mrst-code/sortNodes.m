function n = sortNodes(G)

f = G.cells.faces(:,1);
n = G.faces.nodes(mcolon(G.faces.nodePos(f),G.faces.nodePos(f+1)-1));
s = G.faces.neighbors(f,1) ~= rldecode((1:G.cells.num)', diff(G.cells.facePos),1);

n = reshape(n, 2, []);
n(:,s) = n([2,1], s);
n = n(:);
n = n(1:2:end);

end