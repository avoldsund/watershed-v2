mrstModule add diagnostics

%%
L=50;
G=cartGrid([100, 100],[1000, 1000]);
G=computeGeometry(G);
f=@(x) 0.2*(sin(x(:,1)/L)+sin(x(:,2)/L))+x(:,2)/L;
z=f(G.cells.centroids);
G.nodes.z=f(G.nodes.coords);

Z=reshape(z,G.cartDims);
clf,mesh(Z);
%%
flux=zeros(G.faces.num,1);
ind=all(G.faces.neighbors>0,2);
N=G.faces.neighbors;
flux(ind)=-(z(N(ind,2))-z(N(ind,1)));
state=struct('flux',flux)
rock=struct('poro',ones(G.cells.num,1));

pos=[500,100];
dist=sqrt(sum(bsxfun(@minus,G.cells.centroids,pos).^2,2));
[d,c]=min(dist);
src=addSource([],c,-10);

Gt=G;

tof = computeTimeOfFlight(state, G, rock, 'src', src, ...
   'maxTOF', 1e8, 'reverse', true)
%tof(tof==1e8)=NaN;
clf,plotCellData(G,tof)
%%
to=0;
fr=@(t) exp(-((t-to)/5e6).^2);
th=9e7;
clf,plotCellData(G,(fr(th-tof)))
%%
ths=1e5;
i=0;
for th=linspace(0,ths,100)
    i=i+1;
    ds(i)=sum(G.cells.volumes.*(fr(th+tof)));
end
plot(linspace(0,ths,100),ds)

%