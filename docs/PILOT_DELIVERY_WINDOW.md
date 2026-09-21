# Pilot checkpoint delivery windows

The selected pilot delivery policy permits a copy allowance of 30–660 seconds.
Choose it before launching a new attempt, using observed complete-state transfer,
hashing and acknowledgement costs. A censored partial transfer can inform an
allowance but cannot guarantee completion or qualify throughput.

Each request fixes its deadline to the earlier of the selected copy allowance and
the original job deadline. Acknowledgements at or after that deadline fail, even
when the bytes eventually arrive. The copy allowance cannot renew a request,
extend a job or rental, refresh export health without verified retained bytes, or
grant acceptance to incomplete recording or replay. Existing attempts keep their
original policy, source and deadlines.

Admission still checks the transfer and hashing reserve. Outer progress,
durable-export age, phase duration, watchdog and aggregate budget gates remain in
force. Required checkpoint identity, complete safe-state retention and exact
replay checks are unchanged. Production feasibility must use the actual completed
pilot measurements and include transfer overhead and declared forecast margins.
