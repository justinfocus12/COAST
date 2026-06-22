# Truncated Burgers-Hopf dynamics
\begin{align}
u_t + \frac12\partial_x(u^2) &= 0 \\
u &= \sum_{k=-K}^K\hat{u}_ke^{ikx}\text{ with }\hat{u}_{-k}=\hat{u}k^* \\
u^2 :=w &= \sum_{\ell=-K}^K\sum_{m=-K}^K\hat{u}_\ell\hat{u}_me^{i(\ell+m)x} \\
&= \sum_{k=-K}^K\hat{w}_ke^{ikx} \\
\therefore\hat{w}_k &= \sum_{m=-K}^K\widehat{u}_m\widehat{u}_{k-m}
\end{align}
