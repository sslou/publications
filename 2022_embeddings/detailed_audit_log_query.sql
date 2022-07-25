/*
    PURPOSE: Pulls Access Log Events and Event Details/Mnemonics for the IGNITE Burnout Study
        The query below should be used as part of a SSIS package to pull a given provider's access logs and data around notes and reports viewed/spent time on.

    OUTPUT:
        Datasets in the query output should be saved as .csv files (names denoted in comments above each set) and saved in the following format/hierarchy.

        Working Directory: \\storage1.ris.wustl.edu\lu\Active\ehrlog\IGNITE\data_output_from_SSIS\
        For each provider, create a folder based on their USER_ID (if one does not already exist)
            save .csv's to that folder
        Move to next provider

*/
DECLARE @StartDate AS DATETIME  -- For initial pull, use the provider's enrollment date in the study.  For part 2 of the pull, '4/1/2021'
        , @StopDate AS DATETIME = '4/1/2021'
        , @UserID AS VARCHAR(18)

-- Use only for sample data/testing.  SSIS package should pass in parameters otherwise.
SELECT TOP 1
      @UserID =emp.USER_ID
    , @StartDate = pl.study_enrollment_dttm
FROM  ##IGNITE_Study_Providers pl
    INNER JOIN CLARITY_EMP emp
        ON pl.system_login = emp.SYSTEM_LOGIN
    INNER JOIN CLARITY_SER ser
        ON emp.USER_ID = ser.USER_ID
WHERE
    emp.USER_ID = 'M####'

--SELECT @UserID, @StartDate, @StopDate

-- Build AccessLog Data.  This will be the base data set that you work with in order to fetch mnemonics/details to give better context.

IF OBJECT_ID('tempdb..#accessLog') IS NOT NULL
    DROP TABLE #accessLog
SELECT
    alog.ACCESS_INSTANT
    , aLog.PROCESS_ID
    , alog.ACCESS_TIME
    , alog.USER_ID
    , aLog.WORKSTATION_ID
    , dep.DEPARTMENT_NAME
    , alm.METRIC_ID
    , alm.METRIC_NAME
    --, alm.METRIC_DESC
    , aLog.PAT_ID
    , aLog.CSN
    --, aLog.ACCESS_ACTION_C
INTO #accessLog
FROM ACCESS_LOG aLog
    LEFT OUTER JOIN ACCESS_LOG_METRIC alm
        ON aLog.METRIC_ID = alm.METRIC_ID
    LEFT OUTER JOIN CLARITY_LWS lws
        ON aLog.WORKSTATION_ID = lws.WORKSTN_IDENTIFIER
    LEFT OUTER JOIN CLARITY_DEP dep
        ON lws.DEPARTMENT_ID = dep.DEPARTMENT_ID
    --LEFT OUTER JOIN ZC_ACCESS_ACTION zcAction
    --  ON zcAction.ACCESS_ACTION_C = aLog.ACCESS_ACTION_C
    --LEFT OUTER JOIN ACCESS_LOG_DESC metricDesc
    --  ON alm.METRIC_ID = metricDesc.METRIC_ID
WHERE
    (aLog.ACCESS_TIME >= @StartDate 
    AND aLog.ACCESS_TIME < @StopDate)
    AND aLog.USER_ID = @UserID

-- ACCESS LOG MNEMONICS temp table
-- These details/mnemonics will give more detail/context around a given access log event.  There can be more than one detail/mnemonic to a given access log event.
IF OBJECT_ID('tempdb..#aLogDetailsAndMnem') IS NOT NULL
    DROP TABLE #aLogDetailsAndMnem
SELECT
    aLog.ACCESS_INSTANT
    , aLog.PROCESS_ID
    , aLog.ACCESS_TIME
    , aLogMetric.METRIC_NAME
    , aLog.METRIC_ID
    --, aLogMetric.METRIC_DESC
    , aLog.USER_ID
    , aLog.WORKSTATION_ID
    , aLog.PAT_ID
    , aLog.CSN
    --, aLog.ACCESS_ACTION_C
    --, zcAction.NAME [Access_Action]
    , alogMnem.DATA_MNEMONIC_ID
    , aLogMnem.DATA_DESC
    , aLogMnem.DATA_INI_ITEM
    , aLogDetails.INTEGER_VALUE
    , aLogDetails.STRING_VALUE
INTO #aLogDetailsAndMnem
FROM #accessLog aLog
    LEFT OUTER JOIN ACCESS_LOG_METRIC aLogMetric
        ON aLog.METRIC_ID = aLogMetric.METRIC_ID
    LEFT OUTER JOIN ACC_LOG_DTL_IX aLogDetails
        ON aLog.ACCESS_INSTANT = aLogDetails.ACCESS_INSTANT
        AND aLog.PROCESS_ID = aLogDetails.PROCESS_ID
        --AND alogMnem.DATA_MNEMONIC_ID = aLogDetails.DATA_MNEMONIC_ID
        -- add data_mnemonic id to complete primary key
    LEFT OUTER JOIN ACCESS_LOG_MNEM alogMnem 
        ON aLogDetails.DATA_MNEMONIC_ID = aLogMnem.DATA_MNEMONIC_ID
    --LEFT OUTER JOIN ZC_ACCESS_ACTION zcAction
    --  ON aLog.ACCESS_ACTION_C = zcAction.ACCESS_ACTION_C

-- *******************************************************************************************************************************
-- THE CODE BELOW BUILDS THE EXTRACT DATA SETS FROM THE SOURCE SETS ABOVE

-- *******************************************************************************************************************************


-- raw access log
    -- NAME: access_log_raw.csv
SELECT *
FROM #accessLog a
--WHERE a.[WORKSTATION_ID] IS NULL
--ORDER BY a.ACCESS_INSTANT, a.METRIC_ID 

-- raw access log with details/mnemonics
    -- NAME: access_log_raw_mnemonics.csv
SELECT *
FROM #aLogDetailsAndMnem t

-- joined access log to note data...this generate the Access Log HNO Note View
    -- NAME: access_log_HNO.csv
    -- 22 seconds
SELECT * 
FROM 
    (SELECT *
                FROM #aLogDetailsAndMnem aLogD
                WHERe aLogD.DATA_MNEMONIC_ID = 'HNO'
                    AND aLogD.DATA_INI_ITEM = 'HNO .1'
                    ) aLog
        left outer join (SELECT
                            vNote.NOTE_ID
                            , vNote.DATE_OF_SERVICE_DTTM
                            , zcNoteType.NAME [Note_Type]
                            , zcNoteStatus.NAME [Note_Status]
                            , vNote.AUTHOR_NAME
                            , ser.PROV_TYPE
                            , zcService.NAME [Author_Service]
                            , vNote.AUTHOR_SERVICE_C
                            , vNote.AUTHOR_LOGIN_DEPARTMENT_NAME
                            , rpt15.NAME [rpt15]
                            , rpt16.NAME [rpt16]
                            , rpt17.NAME [rpt17]
                        FROM V_NOTE_CHARACTERISTICS vNote
                            LEFT OUTER JOIN ZC_NOTE_TYPE_IP zcNoteType
                                ON vNote.NOTE_TYPE_C = zcNoteType.TYPE_IP_C
                            LEFT OUTER JOIN ZC_NOTE_STATUS zcNoteStatus
                                ON vNote.NOTE_STATUS_C = zcNoteStatus.NOTE_STATUS_C
                            LEFT OUTER JOIN ZC_CLINICAL_SVC zcService
                                ON vNote.AUTHOR_SERVICE_C = zcService.CLINICAL_SVC_C
                            LEFT OUTER JOIN CLARITY_SER ser
                                ON vNote.AUTHOR_LINKED_PROV_ID = ser.PROV_ID
                            LEFT OUTER JOIN ZC_SER_RPT_GRP_15 rpt15
                                ON ser.RPT_GRP_FIFTEEN_C = rpt15.RPT_GRP_FIFTEEN_C
                            LEFT OUTER JOIN ZC_SER_RPT_GRP_16 rpt16
                                ON ser.RPT_GRP_SIXTEEN_C = rpt16.RPT_GRP_SIXTEEN_C
                            LEFT OUTER JOIN ZC_SER_RPT_GRP_17 rpt17
                                ON ser.RPT_GRP_SEVNTEEN_C = rpt17.RPT_GRP_SEVNTEEN_C
                        WHERE
                            EXISTS (SELECT 1
                                    FROM #aLogDetailsAndMnem aLogD
                                    WHERE
                                        vNote.NOTE_ID = aLogD.STRING_VALUE
                                        AND aLogD.DATA_MNEMONIC_ID = 'HNO'
                                        AND aLogD.DATA_INI_ITEM = 'HNO .1') --added this in case in larger pool other HNO INI ITEMS get used
                        ) notes ON AlOG.STRING_VALUE = notes.NOTE_ID


    
-- Access Log Event/Mnemonics Specific to LRP Records aka Reports Viewed
    --> NAME: access_log_LRP_Reports_View.csv
SELECT
    aLog.ACCESS_INSTANT, aLog.PROCESS_ID, aLog.ACCESS_TIME
    , aLogMetric.METRIC_NAME
    , rd.REPORT_NAME
    --, aLogMetric.METRIC_DESC
    --, aLog.METRIC_ID
    , aLog.USER_ID, aLog.WORKSTATION_ID
    , aLog.PAT_ID
    , aLog.CSN
    , alogMnem.DATA_MNEMONIC_ID
    , aLogMnem.DATA_DESC
    , aLogMnem.DATA_INI_ITEM
    , aLogDetails.INTEGER_VALUE
    , aLogDetails.STRING_VALUE
FROM #accessLog aLog
    LEFT OUTER JOIN ACCESS_LOG_METRIC aLogMetric
        ON aLog.METRIC_ID = aLogMetric.METRIC_ID
    LEFT OUTER JOIN ACC_LOG_DTL_IX aLogDetails
        ON aLog.ACCESS_INSTANT = aLogDetails.ACCESS_INSTANT
        AND aLog.PROCESS_ID = aLogDetails.PROCESS_ID
        -- add data_mnemonic id to complete primary key
    LEFT OUTER JOIN ACCESS_LOG_MNEM alogMnem 
        ON aLogDetails.DATA_MNEMONIC_ID = aLogMnem.DATA_MNEMONIC_ID
    LEFT OUTER JOIN REPORT_DETAILS rd
        ON aLogDetails.INTEGER_VALUE = rd.LRP_ID
        AND alogMnem.DATA_MNEMONIC_ID = 'LRP'
WHERE alogMnem.DATA_MNEMONIC_ID = 'LRP'
ORDER BY aLog.ACCESS_INSTANT, aLog.PROCESS_ID

-- Cleanup...
 DROP TABLE #accessLog, #aLogDetailsAndMnem