clear;
data_uw = importdata("numerical_sol_upwind.txt");
data_lw = importdata("numerical_sol_lax-wendroff.txt");

x = data_uw(:, 1);
y_ex = data_uw(:, 2);

y_uw = data_uw(:, 3);
y_lw = data_lw(:, 3);

figure(1);
plot(x, y_uw, 'rs-', 'LineWidth', 1);
hold on;
plot(x, y_lw, 'b^-', 'LineWidth', 1);
hold on;
plot(x, y_ex, 'k-', 'LineWidth', 1);

xlim([-10.5, 10.5]);
ylim([-1.55, 1.2]);
xlabel('x');
ylabel('u');
legend({'Upwind', 'Lax-Wendroff', 'Exact'}, 'Location','northwest', 'FontSize', 10);
set(gca, 'FontSize', 18);
