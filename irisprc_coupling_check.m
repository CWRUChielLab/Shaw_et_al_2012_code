% Copyright (c) 2011, Peter Thomas
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% * Redistributions of source code must retain the above copyright notice,
%   this list of conditions and the following disclaimer.
%
% * Redistributions in binary form must reproduce the above copyright
%   notice, this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

% code to check prediction of stability/instability for diffusively coupled
% iris systems.

format long

phi=linspace(0, 4, 1000); % phase as defined in Shaw/Young/Chiel/Thomas paper
lam=2; % saddle value
a=.2; % twist parameter -- large
%a=.001; % twist parameter -- small
rho=@(u)u^lam-u+a; % "u" for LC is a root of this equation
uu=fzero(rho,0) % u coordinate for LC igress
ss=uu^lam % s coordinate for LC egress
z1=@(phi)(abs(phi-.5)<.5).*ss.^(1-phi);
z2=@(phi)(abs(phi-1.5)<.5).*uu.^(phi-1);
z3=@(phi)(abs(phi-2.5)<.5).*ss.^(-phi+3)*(-1);
z4=@(phi)(abs(phi-3.5)<.5).*uu.^(phi-3)*(-1);
z=@(phi)z1(phi)+z2(phi)+z3(phi)+z4(phi);

% trajectory
% PJT's first calculation -- incorrect
% x1=@(phi)(abs(phi-.5)<.5).*(-1+uu.^(1-phi));
% x2=@(phi)(abs(phi-1.5)<.5).*(1-ss.^(phi-1));
% x3=@(phi)(abs(phi-2.5)<.5).*(1-uu.^(1-(phi-2)));
% x4=@(phi)(abs(phi-3.5)<.5).*(-1+ss.^(phi-3));
% x=@(phi)x1(phi)+x2(phi)+x3(phi)+x4(phi);

% KMS' calculation -- confirmed by PJT
x1=@(phi)(abs(phi-.5)<.5).*(-1+a/2+ss.^phi);
x2=@(phi)(abs(phi-1.5)<.5).*(-1-a/2+uu.^(2-phi));
x3=@(phi)(abs(phi-2.5)<.5).*(1-a/2-ss.^(phi-2));
x4=@(phi)(abs(phi-3.5)<.5).*(1+a/2-uu.^(4-phi));
x=@(phi)x1(phi)+x2(phi)+x3(phi)+x4(phi);

% dx/dphi
xp1=@(phi)(abs(phi-.5)<.5).*(ss.^phi).*log(ss);
xp2=@(phi)(abs(phi-1.5)<.5).*(-uu.^(2-phi)).*log(uu);
xp3=@(phi)(abs(phi-2.5)<.5).*(-ss.^(phi-2)).*log(ss);
xp4=@(phi)(abs(phi-3.5)<.5).*(uu.^(4-phi)).*log(uu);
xp=@(phi)xp1(phi)+xp2(phi)+xp3(phi)+xp4(phi);

subplot(3,1,1)
plot(phi,z(phi),'LineWidth',3)
ylabel('iPRC','FontSize',20,'FontWeight','bold')
grid on
set(gca,'FontSize',20,'FontWeight','bold')

subplot(3,1,2)
plot(phi,x(phi),'LineWidth',3)
hold on
plot(phi,x(mod(phi+.1,4)),'r','LineWidth',3) 
ylabel('x(\phi)','FontSize',20,'FontWeight','bold')
grid on
set(gca,'FontSize',20,'FontWeight','bold')

subplot(3,1,3)
plot(phi,xp(phi),'LineWidth',3)
xlabel('Phase \phi \in [0, 4)','FontSize',20,'FontWeight','bold')
ylabel('dx/d\phi','FontSize',20,'FontWeight','bold')
hold on
grid on
set(gca,'FontSize',20,'FontWeight','bold')
shg

print -dpdf irisprc_coupling_check.pdf

