function [F] = OKSC(K, F, c, lambda1, lambda2, lambda3, beta, eta)
[~,N]=size(K);
KG=K;

% ADMM parameters
epsilon = 1e-6; maxIter = 20; max_rho = 1e10; rho = 1e-6; %rho = 1e-8;

% Initializations
[U,S,V] = svd(KG);
B = U*sqrt(S)*V';
%fprintf('\nBuild Cmat...');
obj = [];
resid = [];


%Y1 = zeros(N, N);
Y2 = zeros(N, N);
y3 = zeros(1, N);
Y4 = zeros(N, N);
A = zeros(N, N);
E = zeros(N, N);
Y = zeros(N, N);
iter= 0;

    while(iter<maxIter)
        iter = iter+1;
    
        % Update C
        tmp = A + Y2/rho;
        C = max(0, abs(tmp)-lambda1/rho) .* sign(tmp);
        C = C - diag(diag(C));

        % Update A
        
        K = B'*B;
        
        for ij=1:N
            for ji=1:N
                allSP(ji)=(norm(F(ij,:)-F(ji,:)))^2;
            end
        allSP=allSP.*allSP;
        lhs = lambda2*K + rho*eye(N) + rho*ones(N,N);
        X1=ones(N,1)*y3;
        X2=rho*(C-diag(diag(C))+ones(N,N));
        rhs(:,ij) = lambda2*K(:,ij) - Y2(:,ij) - X1(:,ij) + beta/4*allSP' + X2(:,ij);%应该是同时beta/4,故原来的参数�?��乘以2
        %A = inv(lhs(:,ij))*rhs;
        end
        
        A = inv(lhs)*rhs;
        A = A -diag(diag(A));
        A = normcols(A);
        %A = lhs\rhs;            

        % Update B
        %tmp = 0.5*(K - E + KG + (Y1+Y4)/rho);
        tmp = KG - E - (0.5*lambda2*(eye(N)-2*A'+A*A')-Y4)/rho;
        B = solveB(tmp, rho); %low-rank
        %B = solveB_sp(tmp, rho, p); %low-rank-sp

        % Update E
        tmp = KG - B'*B + Y4/rho;
        E = max(0,tmp - lambda3/rho)+min(0,tmp + lambda3/rho);
        %E = solve_l1l2(tmp,lambda3/rho);

        % Update F
        A= (A+A')/2;
        D = diag(sum(A));
        L = D-A;
        [F, temp, ev]=eig1(L, c, 0);
        

        
        leq2 = A - (C - diag(diag(C)));
        leq3 = sum(A) - ones(1,N);
        leq4 = KG - B'*B - E;

        obj(iter) = sum(svd(B)) + lambda1*sum(abs(C(:))) + ...
            0.5*lambda2*trace((eye(N)-2*A+A*A')*(B'*B)) + ...
            lambda3*sum(abs(E(:)));
        resid(iter) = max(max(norm(leq2,'fro'),norm(leq3)), norm(leq4,'fro'));
        
        stpC = max(abs(leq2(:)));
        stpC2 = max(max(abs(leq3)), max(abs(leq4(:))));
        stpC = max(stpC, stpC2);
        if(iter == 1 || mod(iter,50)==0 || stpC<epsilon)
            disp(['iter ' num2str(iter) ',rho=' num2str(rho,'%2.1e') ',stopALM=' num2str(stpC,'%2.3e')]);
        end
        
        if(stpC<epsilon)
            break;
        else
            %Y1 = Y1 + rho*leq1;
            Y2 = Y2 + rho*leq2;
            y3 = y3 + rho*leq3;
            Y4 = Y4 + rho*leq4;
            rho = min(max_rho,rho*eta);
        end
        
    end
   
end