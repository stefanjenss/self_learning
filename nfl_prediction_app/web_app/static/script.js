// Tab functionality
function openTab(tabName) {
    // Hide all tabs
    const tabs = document.getElementsByClassName('tab-content');
    for (let tab of tabs) {
        tab.classList.remove('active');
    }

    // Remove active class from all buttons
    const buttons = document.getElementsByClassName('tab-button');
    for (let button of buttons) {
        button.classList.remove('active');
    }

    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Load predictions
async function loadPredictions() {
    const container = document.getElementById('predictions-container');
    container.innerHTML = '<p class="loading">Loading predictions...</p>';

    try {
        const response = await fetch('/api/predictions');
        const predictions = await response.json();

        if (predictions.error) {
            container.innerHTML = `<p class="error">Error: ${predictions.error}</p>`;
            return;
        }

        let html = '';
        for (let pred of predictions) {
            const confidence = (pred.confidence * 100).toFixed(1);
            const confidenceClass = pred.confidence > 0.65 ? 'confidence-high' :
                                   pred.confidence > 0.55 ? 'confidence-medium' : 'confidence-low';

            const actualScore = pred.home_score !== null ?
                `<div class="actual">Final: ${pred.home_team} ${pred.home_score} - ${pred.away_team} ${pred.away_score}</div>` : '';

            html += `
                <div class="game-card">
                    <h3>Week ${pred.week}, ${pred.season}</h3>
                    <div class="teams">${pred.home_team} vs ${pred.away_team}</div>
                    <div class="prediction ${confidenceClass}">
                        Predicted Winner: ${pred.predicted_winner} (${confidence}% confidence)
                    </div>
                    ${actualScore}
                </div>
            `;
        }

        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<p class="error">Error loading predictions: ${error.message}</p>`;
    }
}

// Run backtest
async function runBacktest() {
    const strategy = document.getElementById('strategy-select').value;
    const bankroll = parseInt(document.getElementById('bankroll-input').value);

    const resultsDiv = document.getElementById('backtest-results');
    resultsDiv.innerHTML = '<p class="loading">Running backtest...</p>';

    try {
        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy, bankroll })
        });

        const results = await response.json();

        if (results.error) {
            resultsDiv.innerHTML = `<p class="error">Error: ${results.error}</p>`;
            return;
        }

        // Display metrics
        const metrics = results.metrics;
        resultsDiv.innerHTML = `
            <h3>${results.strategy} Results</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="value">${metrics.total_bets}</div>
                    <div class="label">Total Bets</div>
                </div>
                <div class="metric-card">
                    <div class="value">${(metrics.win_rate * 100).toFixed(1)}%</div>
                    <div class="label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="value">$${metrics.total_profit.toFixed(2)}</div>
                    <div class="label">Total Profit</div>
                </div>
                <div class="metric-card">
                    <div class="value">${(metrics.roi * 100).toFixed(1)}%</div>
                    <div class="label">ROI</div>
                </div>
                <div class="metric-card">
                    <div class="value">$${metrics.final_bankroll.toFixed(2)}</div>
                    <div class="label">Final Bankroll</div>
                </div>
                <div class="metric-card">
                    <div class="value">${(metrics.max_drawdown * 100).toFixed(1)}%</div>
                    <div class="label">Max Drawdown</div>
                </div>
            </div>
        `;

        // Plot bankroll history
        const trace = {
            y: results.bankroll_history,
            type: 'scatter',
            mode: 'lines',
            name: 'Bankroll',
            line: { color: '#667eea', width: 2 }
        };

        const layout = {
            title: 'Bankroll Over Time',
            xaxis: { title: 'Bet Number' },
            yaxis: { title: 'Bankroll ($)' },
            hovermode: 'closest'
        };

        Plotly.newPlot('bankroll-chart', [trace], layout);

    } catch (error) {
        resultsDiv.innerHTML = `<p class="error">Error running backtest: ${error.message}</p>`;
    }
}

// Compare strategies
async function compareStrategies() {
    const bankroll = parseInt(document.getElementById('compare-bankroll').value);

    const resultsDiv = document.getElementById('comparison-results');
    resultsDiv.innerHTML = '<p class="loading">Comparing strategies...</p>';

    try {
        const response = await fetch('/api/compare_strategies', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bankroll })
        });

        const results = await response.json();

        if (results.error) {
            resultsDiv.innerHTML = `<p class="error">Error: ${results.error}</p>`;
            return;
        }

        // Create comparison table
        let html = '<h3>Strategy Comparison</h3><table>';
        html += '<tr><th>Strategy</th><th>Total Bets</th><th>Win Rate</th><th>ROI</th><th>Final Bankroll</th><th>Max Drawdown</th></tr>';

        for (let result of results) {
            html += `
                <tr>
                    <td><strong>${result.strategy}</strong></td>
                    <td>${result.total_bets}</td>
                    <td>${(result.win_rate * 100).toFixed(1)}%</td>
                    <td>${(result.roi * 100).toFixed(1)}%</td>
                    <td>$${result.final_bankroll.toFixed(2)}</td>
                    <td>${(result.max_drawdown * 100).toFixed(1)}%</td>
                </tr>
            `;
        }

        html += '</table>';
        resultsDiv.innerHTML = html;

        // Create comparison chart
        const trace1 = {
            x: results.map(r => r.strategy),
            y: results.map(r => r.roi * 100),
            type: 'bar',
            name: 'ROI (%)',
            marker: { color: '#667eea' }
        };

        const trace2 = {
            x: results.map(r => r.strategy),
            y: results.map(r => r.win_rate * 100),
            type: 'bar',
            name: 'Win Rate (%)',
            marker: { color: '#764ba2' }
        };

        const layout = {
            title: 'Strategy Performance Comparison',
            barmode: 'group',
            xaxis: { title: 'Strategy' },
            yaxis: { title: 'Percentage (%)' }
        };

        const comparisonDiv = document.createElement('div');
        comparisonDiv.id = 'comparison-chart';
        resultsDiv.appendChild(comparisonDiv);

        Plotly.newPlot('comparison-chart', [trace1, trace2], layout);

    } catch (error) {
        resultsDiv.innerHTML = `<p class="error">Error comparing strategies: ${error.message}</p>`;
    }
}

// Load model info
async function loadModelInfo() {
    const container = document.getElementById('model-info-container');
    container.innerHTML = '<p class="loading">Loading model information...</p>';

    try {
        const response = await fetch('/api/model_info');
        const models = await response.json();

        if (models.error) {
            container.innerHTML = `<p class="error">Error: ${models.error}</p>`;
            return;
        }

        let html = '';
        for (let model of models) {
            html += `
                <div class="game-card">
                    <h3>${model.name}</h3>
                    <p>Status: ${model.fitted ? 'Trained' : 'Not trained'}</p>
                    <p>Number of features: ${model.features}</p>
            `;

            if (model.top_features) {
                html += '<h4>Top 10 Features:</h4><ul>';
                for (let feature of model.top_features) {
                    html += `<li>${feature.feature}: ${feature.importance.toFixed(4)}</li>`;
                }
                html += '</ul>';
            }

            html += '</div>';
        }

        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<p class="error">Error loading model info: ${error.message}</p>`;
    }
}
