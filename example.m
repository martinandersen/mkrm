clc,clear

N = 200;
n = 50;
p = 100;

Phi = randn(N,n);
y = Phi*rand(n,1) + 0.1*randn(N,1);
U = cell(1,p);
for i = 1:p
    U{i} = rand(n,4);
end

options = struct('verbose',1,'ev',true, ...
                 'nv',1e-2, ...
                 'dimension_reduction',true, ...
                 'c_nv',0.0, ...
                 'c_alpha', zeros(p,1));

if exist('mosekopt') == 3
    % Solve with Mosek
    [alpha,nv,info] = mkrm_optimize(y,Phi,U,options);
    
    figure(1), clf
    subplot(2,1,1)
    stem(alpha)
    title('MOSEK')
    subplot(2,1,2)
    plot(0:info.iterations,info.objval,'k.-')
    xlabel('Iterations')
    ylabel('Objective value')
    fprintf(1,'Noise variance: %.2e\n',nv)
    fprintf(1,'Time: %.2f sec.\n\n',sum(info.time))
else
    fprintf(1,'MOSEK not installed.')
end

if exist('sqlp') == 2
    % Solve with SDPT3
    options1 = options;
    options1.solver = 'sdpt3';
    [alpha1,nv1,info1] = mkrm_optimize(y,Phi,U,options1);

    figure(2), clf
    subplot(2,1,1)
    stem(alpha1)
    title('SDPT3')
    subplot(2,1,2)
    plot(0:info1.iterations,info1.objval,'k.-')
    xlabel('Iterations')
    ylabel('Objective value')
    fprintf(1,'Noise variance: %.2e\n',nv1)
    fprintf(1,'Time: %.2f sec.\n\n',sum(info1.time))
else
    fprintf(1,'SDPT3 not installed.')
end

if exist('sedumi') == 2
    % Solve with SeDuMi
    options2 = options;
    options2.solver = 'sedumi';
    [alpha2,nv2,info2] = mkrm_optimize(y,Phi,U,options2);

    figure(3), clf
    subplot(2,1,1)
    stem(alpha2)
    title('SeDuMi')
    subplot(2,1,2)
    plot(0:info2.iterations,info2.objval,'k.-')
    xlabel('Iterations')
    ylabel('Objective value')
    fprintf(1,'Noise variance: %.2e\n',nv2)
    fprintf(1,'Time: %.2f sec.\n\n',sum(info2.time))
else
    fprintf(1,'SeDuMi not installed.')
end
