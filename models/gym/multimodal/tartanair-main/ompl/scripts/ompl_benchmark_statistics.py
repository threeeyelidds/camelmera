#!/usr/bin/env python

######################################################################
# Software License Agreement (BSD License)
#
#  Copyright (c) 2010, Rice University
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Rice University nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
######################################################################

# Author: Mark Moll, Ioan Sucan, Luis G. Torres

from sys import argv, exit
from os.path import basename, splitext
import sqlite3
import datetime
import matplotlib
matplotlib.use('pdf')
from matplotlib import __version__ as matplotlibversion
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from optparse import OptionParser, OptionGroup

def readBenchmarkLog(dbname, filenames):
    """Parse benchmark log files and store the parsed data in a sqlite3 database."""

    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('PRAGMA FOREIGN_KEYS = ON')

    # create all tables if they don't already exist
    c.executescript("""CREATE TABLE IF NOT EXISTS experiments
        (id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR(512),
        totaltime REAL, timelimit REAL, memorylimit REAL, runcount INTEGER,
        hostname VARCHAR(1024), date DATETIME, seed INTEGER, setup TEXT);
        CREATE TABLE IF NOT EXISTS plannerConfigs
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(512) NOT NULL, settings TEXT);
        CREATE TABLE IF NOT EXISTS enums
        (name VARCHAR(512), value INTEGER, description TEXT,
        PRIMARY KEY (name, value));
        CREATE TABLE IF NOT EXISTS runs
        (id INTEGER PRIMARY KEY AUTOINCREMENT, experimentid INTEGER, plannerid INTEGER,
        FOREIGN KEY (experimentid) REFERENCES experiments(id) ON DELETE CASCADE,
        FOREIGN KEY (plannerid) REFERENCES plannerConfigs(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS progress
        (runid INTEGER, time REAL, PRIMARY KEY (runid, time),
        FOREIGN KEY (runid) REFERENCES runs(id) ON DELETE CASCADE)""")

    for filename in filenames:
        print('Processing ' + filename)
        logfile = open(filename,'r')
        expname =  logfile.readline().split()[-1]
        hostname = logfile.readline().split()[-1]
        date = ' '.join(logfile.readline().split()[2:])
        logfile.readline() # skip <<<|
        expsetup = ''
        expline = logfile.readline()
        while not expline.startswith('|>>>'):
            expsetup = expsetup + expline
            expline = logfile.readline()
        rseed = int(logfile.readline().split()[0])
        timelimit = float(logfile.readline().split()[0])
        memorylimit = float(logfile.readline().split()[0])
        nrruns = float(logfile.readline().split()[0])
        totaltime = float(logfile.readline().split()[0])
        numEnums = int(logfile.readline().split()[0])
        for i in range(numEnums):
            enum = logfile.readline()[:-1].split('|')
            c.execute('SELECT * FROM enums WHERE name IS "%s"' % enum[0])
            if c.fetchone()==None:
                for j in range(len(enum)-1):
                    c.execute('INSERT INTO enums VALUES (?,?,?)',
                        (enum[0],j,enum[j+1]))
        c.execute('INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)',
              (None, expname, totaltime, timelimit, memorylimit, nrruns,
              hostname, date, rseed, expsetup) )
        experimentId = c.lastrowid
        numPlanners = int(logfile.readline().split()[0])

        for i in range(numPlanners):
            plannerName = logfile.readline()[:-1]
            print('Parsing data for ' + plannerName)

            # read common data for planner
            numCommon = int(logfile.readline().split()[0])
            settings = ''
            for j in range(numCommon):
                settings = settings + logfile.readline() + ';'

            # find planner id
            c.execute('SELECT id FROM plannerConfigs WHERE (name=? AND settings=?)',
                (plannerName, settings,))
            p = c.fetchone()
            if p==None:
                c.execute('INSERT INTO plannerConfigs VALUES (?,?,?)',
                    (None, plannerName, settings,))
                plannerId = c.lastrowid
            else:
                plannerId = p[0]

            # get current column names
            c.execute('PRAGMA table_info(runs)')
            columnNames = [col[1] for col in c.fetchall()]

            # read properties and add columns as necessary
            numProperties = int(logfile.readline().split()[0])
            propertyNames = ['experimentid', 'plannerid']
            for j in range(numProperties):
                field = logfile.readline().split()
                propertyType = field[-1]
                propertyName = '_'.join(field[:-1])
                if propertyName not in columnNames:
                    c.execute('ALTER TABLE runs ADD %s %s' % (propertyName, propertyType))
                propertyNames.append(propertyName)
            # read measurements
            insertFmtStr = 'INSERT INTO runs (' + ','.join(propertyNames) + \
                ') VALUES (' + ','.join('?'*len(propertyNames)) + ')'
            numRuns = int(logfile.readline().split()[0])
            runIds = []
            for j in range(numRuns):
                values = tuple([experimentId, plannerId] + [None if len(x)==0 else x
                    for x in logfile.readline().split('; ')[:-1]])
                c.execute(insertFmtStr, values)
                # extract primary key of each run row so we can reference them
                # in the planner progress data table if needed
                runIds.append(c.lastrowid)

            nextLine = logfile.readline().strip()

            # read planner progress data if it's supplied
            if nextLine != '.':
                # get current column names
                c.execute('PRAGMA table_info(progress)')
                columnNames = [col[1] for col in c.fetchall()]

                # read progress properties and add columns as necesary
                numProgressProperties = int(nextLine.split()[0])
                progressPropertyNames = ['runid']
                for i in range(numProgressProperties):
                    field = logfile.readline().split()
                    progressPropertyType = field[-1]
                    progressPropertyName = "_".join(field[:-1])
                    if progressPropertyName not in columnNames:
                        c.execute('ALTER TABLE progress ADD %s %s' %
                            (progressPropertyName, progressPropertyType))
                    progressPropertyNames.append(progressPropertyName)
                # read progress measurements
                insertFmtStr = 'INSERT INTO progress (' + \
                    ','.join(progressPropertyNames) + ') VALUES (' + \
                    ','.join('?'*len(progressPropertyNames)) + ')'
                numRuns = int(logfile.readline().split()[0])
                for j in range(numRuns):
                    dataSeries = logfile.readline().split(';')[:-1]
                    for dataSample in dataSeries:
                        values = tuple([runIds[j]] + \
                            [None if x == 'nan' else x for x in dataSample.split(',')[:-1]])
                        c.execute(insertFmtStr, values)

                logfile.readline()
        logfile.close()
    conn.commit()
    c.close()

def plotAttribute(cur, planners, attribute, typename):
    """Create a plot for a particular attribute. It will include data for
    all planners that have data for this attribute."""
    plt.clf()
    ax = plt.gca()
    labels = []
    measurements = []
    nanCounts = []
    if typename == 'ENUM':
        cur.execute('SELECT description FROM enums where name IS "%s"' % attribute)
        descriptions = [ t[0] for t in cur.fetchall() ]
        numValues = len(descriptions)
    for planner in planners:
        cur.execute('SELECT %s FROM runs WHERE plannerid = %s AND %s IS NOT NULL' \
            % (attribute, planner[0], attribute))
        measurement = [ 0 if np.isinf(t[0]) else t[0] for t in cur.fetchall() ]
        if len(measurement) > 0:
            cur.execute('SELECT count(*) FROM runs WHERE plannerid = %s AND %s IS NULL' \
                % (planner[0], attribute))
            nanCounts.append(cur.fetchone()[0])
            labels.append(planner[1])
            if typename == 'ENUM':
                scale = 100. / len(measurement)
                measurements.append([measurement.count(i)*scale for i in range(numValues)])
            else:
                measurements.append(measurement)

    if typename == 'ENUM':
        width = .5
        measurements = np.transpose(np.vstack(measurements))
        colsum = np.sum(measurements, axis=1)
        rows = np.where(colsum != 0)[0]
        heights = np.zeros((1,measurements.shape[1]))
        ind = range(measurements.shape[1])
        legend_labels = []
        for i in rows:
            plt.bar(ind, measurements[i], width, bottom=heights[0],
                color=matplotlib.cm.hot(int(floor(i*256/numValues))),
                label=descriptions[i])
            heights = heights + measurements[i]
        xtickNames = plt.xticks([x+width/2. for x in ind], labels, rotation=30)
        ax.set_ylabel(attribute.replace('_',' ') + ' (%)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        props = matplotlib.font_manager.FontProperties()
        props.set_size('small')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = props)
    elif typename == 'BOOLEAN':
        width = .5
        measurementsPercentage = [sum(m) * 100. / len(m) for m in measurements]
        ind = range(len(measurements))
        plt.bar(ind, measurementsPercentage, width)
        xtickNames = plt.xticks([x + width / 2. for x in ind], labels, rotation=30)
        ax.set_ylabel(attribute.replace('_',' ') + ' (%)')
    else:
        if int(matplotlibversion.split('.')[0])<1:
            plt.boxplot(measurements, notch=0, sym='k+', vert=1, whis=1.5)
        else:
            plt.boxplot(measurements, notch=0, sym='k+', vert=1, whis=1.5, bootstrap=1000)
        ax.set_ylabel(attribute.replace('_',' '))
        xtickNames = plt.setp(ax,xticklabels=labels)
        plt.setp(xtickNames, rotation=25)
    ax.set_xlabel('Motion planning algorithm')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    if max(nanCounts)>0:
        maxy = max([max(y) for y in measurements])
        for i in range(len(labels)):
            x = i+width/2 if typename=='BOOLEAN' else i+1
            ax.text(x, .95*maxy, str(nanCounts[i]), horizontalalignment='center', size='small')
    plt.show()

def plotProgressAttribute(cur, planners, attribute):
    """Plot data for a single planner progress attribute. Will create an
average time-plot with error bars of the attribute over all runs for
each planner."""

    import numpy.ma as ma

    plt.clf()
    ax = plt.gca()
    ax.set_xlabel('time (s)')
    ax.set_ylabel(attribute.replace('_',' '))
    plannerNames = []
    for planner in planners:
        cur.execute("""SELECT count(progress.%s) FROM progress INNER JOIN runs
            ON progress.runid = runs.id AND runs.plannerid=%s
            AND progress.%s IS NOT NULL""" \
            % (attribute, planner[0], attribute))
        if cur.fetchone()[0] > 0:
            plannerNames.append(planner[1])
            cur.execute("""SELECT DISTINCT progress.runid FROM progress INNER JOIN runs
            WHERE progress.runid=runs.id AND runs.plannerid=?""", (planner[0],))
            runids = [t[0] for t in cur.fetchall()]
            timeTable = []
            dataTable = []
            for r in runids:
                # Select data for given run
                cur.execute('SELECT time, %s FROM progress WHERE runid = %s ORDER BY time' % (attribute,r))
                (time, data) = zip(*(cur.fetchall()))
                timeTable.append(time)
                dataTable.append(data)
            # It's conceivable that the sampling process may have
            # generated more samples for one run than another; in this
            # case, truncate all data series to length of shortest
            # one.
            fewestSamples = min(len(time[:]) for time in timeTable)
            times = np.array(timeTable[0][:fewestSamples])
            dataArrays = np.array([data[:fewestSamples] for data in dataTable])
            filteredData = ma.masked_array(dataArrays, np.equal(dataArrays, None), dtype=float)

            means = np.mean(filteredData, axis=0)
            stddevs = np.std(filteredData, axis=0, ddof=1)

            # plot average with error bars
            #plt.errorbar(times, means, yerr=2*stddevs, errorevery=max(1, len(times) // 20))
            ax.legend(plannerNames)
    plt.show()

def plotStatistics(dbname, fname):
    """Create a PDF file with box plots for all attributes."""
    print("Generating plots...")
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('PRAGMA FOREIGN_KEYS = ON')
    c.execute('SELECT id, name FROM plannerConfigs')
    planners = [(t[0],t[1].replace('geometric_','').replace('control_',''))
        for t in c.fetchall()]
    c.execute('PRAGMA table_info(runs)')
    colInfo = c.fetchall()[3:]

    pp = PdfPages(fname)
    for col in colInfo:
        if col[2] == 'BOOLEAN' or col[2] == 'ENUM' or \
           col[2] == 'INTEGER' or col[2] == 'REAL':
            plotAttribute(c, planners, col[1], col[2])
            pp.savefig(plt.gcf())
    plt.clf()

    c.execute('PRAGMA table_info(progress)')
    colInfo = c.fetchall()[2:]
    for col in colInfo:
        plotProgressAttribute(c, planners, col[1])
        pp.savefig(plt.gcf())
    plt.clf()

    pagey = 0.9
    pagex = 0.06
    c.execute("""SELECT id, name, timelimit, memorylimit FROM experiments""")
    experiments = c.fetchall()
    for experiment in experiments:
        c.execute("""SELECT count(*) FROM runs WHERE runs.experimentid = %d
            GROUP BY runs.plannerid""" % experiment[0])
        numRuns = [run[0] for run in c.fetchall()]
        numRuns = numRuns[0] if len(set(numRuns)) == 1 else ','.join(numRuns)

        plt.figtext(pagex, pagey, 'Experiment "%s"' % experiment[1])
        plt.figtext(pagex, pagey-0.05, 'Number of averaged runs: %d' % numRuns)
        plt.figtext(pagex, pagey-0.10, "Time limit per run: %g seconds" % experiment[2])
        plt.figtext(pagex, pagey-0.15, "Memory limit per run: %g MB" % experiment[3])
        pagey -= 0.22
    plt.show()
    pp.savefig(plt.gcf())
    pp.close()

def saveAsMysql(dbname, mysqldump):
    # See http://stackoverflow.com/questions/1067060/perl-to-python
    import re
    print("Saving as MySQL dump file...")

    conn = sqlite3.connect(dbname)
    mysqldump = open(mysqldump,'w')

    # make sure all tables are dropped in an order that keepd foreign keys valid
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [ str(t[0]) for t in c.fetchall() ]
    c.close()
    last = ['experiments', 'planner_configs']
    for table in table_names:
        if table.startswith("sqlite"):
            continue
        if not table in last:
            mysqldump.write("DROP TABLE IF EXISTS `%s`;\n" % table)
    for table in last:
        if table in table_names:
            mysqldump.write("DROP TABLE IF EXISTS `%s`;\n" % table)

    for line in conn.iterdump():
        process = False
        for nope in ('BEGIN TRANSACTION','COMMIT',
            'sqlite_sequence','CREATE UNIQUE INDEX', 'CREATE VIEW'):
            if nope in line: break
        else:
            process = True
        if not process: continue
        line = re.sub(r"[\n\r\t ]+", " ", line)
        m = re.search('CREATE TABLE ([a-zA-Z0-9_]*)(.*)', line)
        if m:
            name, sub = m.groups()
            sub = sub.replace('"','`')
            line = '''CREATE TABLE IF NOT EXISTS %(name)s%(sub)s'''
            line = line % dict(name=name, sub=sub)
            # make sure we use an engine that supports foreign keys
            line = line.rstrip("\n\t ;") + " ENGINE = InnoDB;\n"
        else:
            m = re.search('INSERT INTO "([a-zA-Z0-9_]*)"(.*)', line)
            if m:
                line = 'INSERT INTO %s%s\n' % m.groups()
                line = line.replace('"', r'\"')
                line = line.replace('"', "'")

        line = re.sub(r"([^'])'t'(.)", "\\1THIS_IS_TRUE\\2", line)
        line = line.replace('THIS_IS_TRUE', '1')
        line = re.sub(r"([^'])'f'(.)", "\\1THIS_IS_FALSE\\2", line)
        line = line.replace('THIS_IS_FALSE', '0')
        line = line.replace('AUTOINCREMENT', 'AUTO_INCREMENT')
        mysqldump.write(line)
    mysqldump.close()

def computeViews(dbname):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('PRAGMA FOREIGN_KEYS = ON')
    s0 = """SELECT plannerid, plannerConfigs.name AS plannerName, experimentid, solved, time + simplification_time AS total_time
        FROM plannerConfigs INNER JOIN experiments INNER JOIN runs
        ON plannerConfigs.id=runs.plannerid AND experiments.id=runs.experimentid"""
    s1 = """SELECT plannerid, plannerName, experimentid, AVG(solved) AS avg_solved, AVG(total_time) AS avg_total_time
        FROM (%s) GROUP BY plannerid, experimentid""" % s0
    s2 = """SELECT plannerid, experimentid, MIN(avg_solved) AS avg_solved, avg_total_time
        FROM (%s) GROUP BY plannerName, experimentid ORDER BY avg_solved DESC, avg_total_time ASC""" % s1
    c.execute('DROP VIEW IF EXISTS bestPlannerConfigsPerExperiment')
    c.execute('CREATE VIEW IF NOT EXISTS bestPlannerConfigsPerExperiment AS %s' % s2)

    s1 = """SELECT plannerid, plannerName, AVG(solved) AS avg_solved, AVG(total_time) AS avg_total_time
        FROM (%s) GROUP BY plannerid""" % s0
    s2 = """SELECT plannerid, MIN(avg_solved) AS avg_solved, avg_total_time
        FROM (%s) GROUP BY plannerName ORDER BY avg_solved DESC, avg_total_time ASC""" % s1
    c.execute('DROP VIEW IF EXISTS bestPlannerConfigs')
    c.execute('CREATE VIEW IF NOT EXISTS bestPlannerConfigs AS %s' % s2)

    conn.commit()
    c.close()

if __name__ == "__main__":
    usage = """%prog [options] [<benchmark.log> ...]"""
    parser = OptionParser(usage)
    parser.add_option("-d", "--database", dest="dbname", default="benchmark.db",
        help="Filename of benchmark database [default: %default]")
    parser.add_option("-v", "--view", action="store_true", dest="view", default=False,
        help="Compute the views for best planner configurations")
    parser.add_option("-p", "--plot", dest="plot", default=None,
        help="Create a PDF of plots")
    parser.add_option("-m", "--mysql", dest="mysqldb", default=None,
        help="Save SQLite3 database as a MySQL dump file")
    (options, args) = parser.parse_args()

    if len(args)>0:
        readBenchmarkLog(options.dbname, args)
        # If we update the database, we recompute the views as well
        options.view = True

    if options.view:
        computeViews(options.dbname)

    if options.plot:
        plotStatistics(options.dbname, options.plot)

    if options.mysqldb:
        saveAsMysql(options.dbname, options.mysqldb)
