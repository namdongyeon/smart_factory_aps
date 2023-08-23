
select * from new_pred2;


-- 레시피 
select * from recipe1;

drop table new_re;
create table new_re as
select r.*, ROW_NUMBER() OVER (PARTITION BY 제품명, 제품코드, 원자재코드 ORDER BY 투입지시비율 DESC) as rn
from (select
		제품명,
	    제품코드,
	    LOT번호,
	    제품bom차수,
	    원자재투입순번,
	    원자재코드,
	    원자재명,
	    투입지시중량,
	    round(투입지시비율, 2) as 투입지시비율,
	    sitebusname,
	    region
	FROM (
	    SELECT
	        p.prodname as 제품명,
	        r.제품코드,
	        r.LOT번호, 
	        r.제품bom차수,
	        r.원자재투입순번,
	        r.원자재코드,
	        r.원자재명,
	        r.투입지시중량,
	        r.투입지시비율,
	        p.sitebusname,
	        p.region
	    FROM recipe1 r
	    JOIN prodstd3 p ON r.제품코드 = p.제품코드 
		) AS subquery) as r
ORDER BY 제품코드, 제품bom차수, 원자재투입순번;

drop table new_re2;
create table new_re2 as
select 제품명, 원자재코드, round(sum(sum(투입지시중량)) over (partition by 제품명), 2) as 총중량, 원자재명, round(sum(투입지시중량), 2) as 투입지시중량, 
	round(sum(투입지시중량) / (sum(sum(투입지시중량)) over (partition by 제품명)), 6) as 비율,
	 case 
		when regexp_replace(regexp_replace(원자재명, '[가-힣]+', ''),'\\(.*', '') in (select PRODNAME from prodstd3)
			then '제품' else '0' end as 제품여부 
from new_re
group by 제품명, 원자재코드, 원자재명;

select * from new_re2;

-- dataset0 -> 주어진 2년치 데이터를 수요예측을 위한 테이블로 가공

select * from dataset;

drop table dataset0;
create table dataset0 as
SELECT
    week_table.SOLDDATE,
    prodname_table.prodname as PRODNAME,
    COALESCE(SUM(ORDER_QUANT), 0) AS ORDER_QUANT,
    COALESCE(SUM(SOLD_QUANT), 0) AS SOLD_QUANT
FROM (
    SELECT DISTINCT solddate
    FROM dataset
) AS week_table
CROSS JOIN (
    SELECT DISTINCT prodname
    FROM dataset
) AS prodname_table
LEFT JOIN dataset ON week_table.solddate = dataset.SOLDDATE
                 AND prodname_table.prodname = dataset.prodname
GROUP BY week_table.solddate, prodname_table.prodname;

select * from dataset0;

-- new_pred2 -> sarima 예측을 통해 각 제품의 수요 예측값을 추출한 데이터 프레임을 테이블로 업데이트 해서 제품명을 추가하는 등 간단한 가공후에 
-- 생산예측 모델에 적용될수 있도록 외부데이터를 피처로 활용 하는 형태의 테이블로 가공

select * from ae;
select * from csa4000;
select * from csa4000_pca;
select * from csa5000;
select * from pema_500fr;
select * from pema_580fx;
select * from pema_cr1000;
select * from pema_csa5000;
select * from pema_hr1000;
select * from pema_hr1000s;
select * from pema_hr1500;
select * from pema_pcm2000;
select * from pema_pcm2000b;
select * from pema_pcr3000e;
select * from pema_pcr3000n;
select * from pema_pr1000;
select * from pema_pr2000;
select * from pema_sn400;
select * from pema_sp1000;
select * from pema_spr;
select * from pema_sr2000;
select * from pema_sr2000a;
select * from pema_sr3000f;
select * from pema_sr5000f;
select * from pr1000;
select * from sre_110;
select * from sre_200;

drop table ae1;
create table ae1 as
select * from ae;

drop table csa40001;
create table csa40001 as
select * from csa4000;

drop table csa4000_pca1;
create table csa4000_pca1 as
select * from csa4000_pca;

drop table csa50001;
create table csa50001 as
select * from csa5000;

drop table pema_500fr1;
create table pema_500fr1 as
select * from pema_500fr;

drop table pema_580fx1;
create table pema_580fx1 as
select * from pema_580fx;

drop table pema_cr10001;
create table pema_cr10001 as
select * from pema_cr1000;

drop table pema_csa50001;
create table pema_csa50001 as
select * from pema_csa5000;

drop table pema_hr10001;
create table pema_hr10001 as
select * from pema_hr1000;

drop table pema_hr1000s1;
create table pema_hr1000s1 as
select * from pema_hr1000s;

drop table pema_hr15001;
create table pema_hr15001 as
select * from pema_hr1500;

drop table pema_pcm20001;
create table pema_pcm20001 as
select * from pema_pcm2000;

drop table pema_pcm2000b1;
create table pema_pcm2000b1 as
select * from pema_pcm2000b;

drop table pema_pcr3000e1;
create table pema_pcr3000e1 as
select * from pema_pcr3000e;

drop table pema_pcr3000n1;
create table pema_pcr3000n1 as
select * from pema_pcr3000n;

drop table pema_pr10001;
create table pema_pr10001 as
select * from pema_pr1000;

drop table pema_pr20001;
create table pema_pr20001 as
select * from pema_pr2000;

drop table pema_sn4001;
create table pema_sn4001 as
select * from pema_sn400;

drop table pema_sp10001;
create table pema_sp10001 as
select * from pema_sp1000;

drop table pema_spr1;
create table pema_spr1 as
select * from pema_spr;

drop table pema_sr20001;
create table pema_sr20001 as
select * from pema_sr2000;

drop table pema_sr2000a1;
create table pema_sr2000a1 as
select * from pema_sr2000a;

drop table pema_sr3000f1;
create table pema_sr3000f1 as
select * from pema_sr3000f;

drop table pema_sr5000f1;
create table pema_sr5000f1 as
select * from pema_sr5000f;

drop table pr10001;
create table pr10001 as
select * from pr1000;

drop table sre_1101;
create table sre_1101 as
select * from sre_110;

drop table sre_2001;
create table sre_2001 as
select * from sre_200;

alter table ae1
add prodname varchar(33) default 'AE';

alter table csa40001
add prodname varchar(33) default 'csa4000';

alter table csa4000_pca1
add prodname varchar(33) default 'csa4000(pca)';

alter table csa50001
add prodname varchar(33) default 'csa5000';

alter table pema_500fr1
add prodname varchar(33) default 'pema-500fr';

alter table pema_580fx1
add prodname varchar(33) default 'pema-580fx';

alter table pema_cr10001
add prodname varchar(33) default 'pema-cr1000';

alter table pema_csa50001
add prodname varchar(33) default 'pema-csa5000';

alter table pema_hr10001
add prodname varchar(33) default 'pema-hr1000';

alter table pema_hr1000s1
add prodname varchar(33) default 'pema-hr1000s';

alter table pema_hr15001
add prodname varchar(33) default 'pema-hr1500';

alter table pema_pcm20001
add prodname varchar(33) default 'pema-pcm2000';

alter table pema_pcm2000b1
add prodname varchar(33) default 'pema-pcm2000b';

alter table pema_pcr3000e1
add prodname varchar(33) default 'pema-pcr3000e';

alter table pema_pcr3000n1
add prodname varchar(33) default 'pema-pcr3000n';

alter table pema_pr10001
add prodname varchar(33) default 'pema-pr1000';

alter table pema_pr20001
add prodname varchar(33) default 'pema-pr2000';

alter table pema_sn4001
add prodname varchar(33) default 'pema-sn400';

alter table pema_sp10001
add prodname varchar(33) default 'pema-sp1000';

alter table pema_spr1
add prodname varchar(33) default 'pema-spr';

alter table pema_sr20001
add prodname varchar(33) default 'pema-sr2000';

alter table pema_sr2000a1
add prodname varchar(33) default 'pema-sr2000a';

alter table pema_sr3000f1
add prodname varchar(33) default 'pema-sr3000f';

alter table pema_sr5000f1
add prodname varchar(33) default 'pema-sr5000f';

alter table pr10001
add prodname varchar(33) default 'pr1000';

alter table sre_1101
add prodname varchar(33) default 'sre-110';

alter table sre_2001
add prodname varchar(33) default 'sre-200';


select * from ae1;
select * from csa40001;
select * from csa4000_pca1;
select * from csa50001;
select * from pema_500fr1;
select * from pema_580fx1;
select * from pema_cr10001;
select * from pema_csa50001;
select * from pema_hr10001;
select * from pema_hr1000s1;
select * from pema_hr15001;
select * from pema_pcm20001;
select * from pema_pcm2000b1;
select * from pema_pcr3000e1;
select * from pema_pcr3000n1;
select * from pema_pr10001;
select * from pema_pr20001;
select * from pema_sn4001;
select * from pema_sp10001;
select * from pema_spr1;
select * from pema_sr20001;
select * from pema_sr2000a1;
select * from pema_sr3000f1;
select * from pema_sr5000f1;
select * from pr10001;
select * from sre_1101;
select * from sre_2001;

drop table new_pred;
create table new_pred as
select * from ae1
union 
select * from csa40001
union 
select * from csa4000_pca1
union 
select * from csa50001
union 
select * from pema_500fr1
union 
select * from pema_580fx1
union 
select * from pema_cr10001
union
select * from pema_csa50001
union 
select * from pema_hr10001
union 
select * from pema_hr1000s1
union 
select * from pema_hr15001
union 
select * from pema_pcm20001
union 
select * from pema_pcm2000b1
union 
select * from pema_pcr3000e1
union 
select * from pema_pcr3000n1
union 
select * from pema_pr10001
union 
select * from pema_pr20001
union 
select * from pema_sn4001
union 
select * from pema_sp10001
union 
select * from pema_spr1
union 
select * from pema_sr20001
union 
select * from pema_sr2000a1
union 
select * from pema_sr3000f1
union 
select * from pema_sr5000f1
union 
select * from pr10001
union 
select * from sre_1101
union 
select * from sre_2001;

select * from new_pred;

drop table new_pred1;
create table new_pred1 as
select * from new_pred;

select * from new_pred1;

select * from wea;

drop table wea1;
create table wea1 as
select 
	distinct DATE_SUB(tm , INTERVAL (DAYOFWEEK(tm) + 5) % 7 DAY) AS week, 
	round(avg(ta_avg) over (ORDER BY DATE_SUB(tm , INTERVAL (DAYOFWEEK(tm) + 5) % 7 DAY)), 2) as tem_avg, 
	round(avg(avgrhm) over (ORDER BY DATE_SUB(tm, INTERVAL (DAYOFWEEK(tm) + 5) % 7 DAY)), 2) as hum_avg
from wea
where DATE_SUB(tm , INTERVAL (DAYOFWEEK(tm) + 5) % 7 DAY) >= '2021-04-26';

select * from wea1;

update new_pred1 
set prodname = upper(prodname); 

ALTER TABLE new_pred1
add (tem_avg FLOAT, 
	hum_avg FLOAT, 
	SOLD_QUANT FLOAT, 
	국내건설수주액 INT, 
	국내기성액 INT, 
	rn INT);

update new_pred1
set 국내건설수주액 =
	case 
		when substr(week, 1, 7) = '2021-04' then 17774868
		when substr(week, 1, 7) = '2021-05' then 15116048
		when substr(week, 1, 7) = '2021-06' then 18152995
		when substr(week, 1, 7) = '2021-07' then 14541827
	end,
	국내기성액 = 
	case 
		when substr(week, 1, 7) = '2021-04' then 11514860
		when substr(week, 1, 7) = '2021-05' then 11154135
		when substr(week, 1, 7) = '2021-06' then 13252426
		when substr(week, 1, 7) = '2021-07' then 10992998
	end,
	tem_avg = 
    (SELECT tem_avg FROM wea1 WHERE wea1.week = new_pred1.week),
    hum_avg = 
    (SELECT hum_avg FROM wea1 WHERE wea1.week = new_pred1.week),
	rn = 
	case 
		when substr(week, 1, 7) = '2021-04' then 130
		when substr(week, 1, 7) = '2021-05' then 131
		when substr(week, 1, 7) = '2021-06' then 132
		when substr(week, 1, 7) = '2021-07' then 133
	end;

drop table new_pred2;
create table new_pred2 as
select week, prodname, 
	round(tem_avg, 2) as tem_avg, round(hum_avg, 2) as hum_avg,
	round(order_quant, 2) as order_quant, 
	sold_quant, 국내건설수주액, 국내기성액, rn
from new_pred1
order by week, prodname;

select * from new_pred2;

-- pred_prot0 -> 생산예측을 위한 테이블로 일별 데이터를 주별 단위로 모든 제품에 대한 수주량과 평균 온도, 평균 습도, 그리고 닫별로 변하는 건설수주액과 기성액으로 가공된 테이블

drop table lstm_prot;
create table lstm_prot as
select DATE_SUB(substr(SOLDDATE, 1, 10), INTERVAL (DAYOFWEEK(substr(SOLDDATE, 1, 10)) + 5) % 7 DAY) AS week, 
	substr(SOLDDATE, 1, 10) as solddate , PRODNAME , round(D_DAY_2_TEM, 1) as D_DAY_2_TEM , round(D_DAY_2_HUM, 1) as D_DAY_2_HUM , 국내건설수주액, 국내기성액
from lstm
order by week, PRODNAME  ;

select * from lstm_prot;

drop table pred_prot;
create table pred_prot as
SELECT
    week_table.week as WEEK,
    prodname_table.prodname as PRODNAME,
	COALESCE(SUM(ORDER_QUANT), 0) AS ORDER_QUANT,
    COALESCE(SUM(SOLD_QUANT), 0) AS SOLD_QUANT,
    rn
FROM (
    SELECT DISTINCT DATE_SUB(d.SOLDDATE, INTERVAL (DAYOFWEEK(d.SOLDDATE) + 5) % 7 DAY) AS week,
	DENSE_RANK() OVER (ORDER BY DATE_SUB(d.SOLDDATE, INTERVAL (DAYOFWEEK(d.SOLDDATE) + 5) % 7 DAY)) as rn
    FROM dataset d 
) AS week_table
CROSS JOIN (
    SELECT DISTINCT prodname
    FROM dataset
) AS prodname_table
LEFT JOIN dataset ON week_table.week = DATE_SUB(dataset.SOLDDATE, INTERVAL (DAYOFWEEK(dataset.SOLDDATE) + 5) % 7 DAY)
                 AND prodname_table.prodname = dataset.prodname
GROUP BY week_table.week, prodname_table.prodname;

select * from pred_prot;
select * from pred;
select *, avg(D_DAY_2_TEM) from lstm_prot group by week;
select * from lstm_prot1;

alter table pred_prot
add (TEM_AVG float,
	HUM_AVG float,
	국내건설수주액 int(33),
	국내기성액 int(33))

UPDATE pred_prot
JOIN (
    SELECT 
        DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY) AS WEEK, 
        ROUND(AVG(D_DAY_2_TEM) OVER (ORDER BY DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY)), 2) AS tem_avg,
        ROUND(AVG(D_DAY_2_hum) OVER (ORDER BY DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY)), 2) AS hum_avg
    FROM dataset
) AS d
ON pred_prot.week = d.week
SET pred_prot.tem_avg = d.tem_avg,
    pred_prot.hum_avg = d.hum_avg;
   
UPDATE pred_prot
JOIN (
    select 
	week,
	case 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2018-11' and 
			substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2018-12' 
		then 11924951 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2018-12' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-01' 
		then 21742503 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-01' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-02' 
		then 9354316 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-02' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-03' 
		then 7769343 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-03' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-04' 
		then 16252241 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-04' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-05' 
		then 12669551 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-05' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-06' 
		then 10025314 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-06' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-07' 
		then 11286381 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-07' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-08' 
		then 8176838 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-08' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-09' 
		then 8400443 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-09' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-10' 
		then 14366040 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-10' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-11' 
		then 15256861 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-11' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-12' 
		then 13752803 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-12' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-01'
		then 26932542 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-01' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-02' 
		then 10102160 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-02' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-03' 
		then 10780581 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-03' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-04' 
		then 12168031 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-04' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-05' 
		then 8620626 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-05' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-06' 
		then 13766761 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-06' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-07' 
		then 20616667 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-07' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-08' 
		then 14376152 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-08' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-09' 
		then 12679833 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-09' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-10' 
		then 15474662 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-10' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-11' 
		then 14002749 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-11' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-12' 
		then 18059105 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-12' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-01'
		then 29218726 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-01' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-02' 
		then 12915638 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-02' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-03' 
		then 12230562 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-03' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-04' 
		then 16652501 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-04' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-05' 
		then 17774868 else 국내건설수주액
	end as 국내건설수주액, 
	case 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2018-11' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2018-12' 
		then 11419899 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2018-12' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-01' 
		then 13836199 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-01' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-02' 
		then 10473761 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-02' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-03' 
		then 9346282 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-03' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-04' 
		then 11885399 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-04' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-05' 
		then 11532995 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-05' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-06' 
		then 11769196 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-06' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-07' 
		then 13290092 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-07' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-08' 
		then 11111789 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-08' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-09' 
		then 11074985 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-09' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-10' 
		then 10846175 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-10' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-11' 
		then 11424313 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-11' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2019-12' 
		then 11582021 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2019-12' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-01'
		then 14974433 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-01' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-02' 
		then 10089997 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-02' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-03' 
		then 10078824 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-03' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-04' 
		then 12480302 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-04' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-05' 
		then 11349314 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-05' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-06' 
		then 11232028 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-06' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-07' 
		then 13068203 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-07' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-08' 
		then 11049793 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-08' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-09' 
		then 10116690 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-09' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-10' 
		then 11557750 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-10' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-11' 
		then 10589025 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-11' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2020-12' 
		then 11979692 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2020-12' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-01'
		then 14921666 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-01' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-02' 
		then 9457803 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-02' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-03' 
		then 9404361 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-03' and
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-04' 
		then 12145097 
		when substr(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), 1, 7) = '2021-04' and 
		substr(DATE_ADD(DATE_SUB(solddate, INTERVAL (DAYOFWEEK(solddate) + 5) % 7 DAY), INTERVAL 6 DAY), 1, 7) = '2021-05' 
		then 11514860 else 국내기성액
	end as 국내기성액, 
	DENSE_RANK() OVER (ORDER BY DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY)) as rn
from lstm_prot
) AS l
ON pred_prot.week = l.week
SET pred_prot.국내건설수주액 = l.국내건설수주액,
    pred_prot.국내기성액 = l.국내기성액 ;

select * from pred_prot;

drop table pred_prot0;
create table pred_prot0 as
select 
	week,
	prodname,
	round(tem_avg, 2) as tem_avg,
	round(hum_avg, 2) as hum_avg,
	order_quant,
	sold_quant,
	국내건설수주액,
	국내기성액,
	rn
from pred_prot;

select * from pred_prot0;

-- vscode 실행을 통해 업데이트 되는 테이블과 각 과정이 잘 작동되는지 또는 수정이 필요한지 확인을 위한 프로시저

-- *데이터 입력
delete from dataset00 
where SOLDDATE > '2021-04-24';

drop table dataset00;
create table dataset00 as
SELECT
    week_table.SOLDDATE,
    prodname_table.prodname as PRODNAME,
    COALESCE(SUM(ORDER_QUANT), 0) AS ORDER_QUANT,
    COALESCE(SUM(SOLD_QUANT), 0) AS SOLD_QUANT
FROM (
    SELECT DISTINCT solddate
    FROM dataset
) AS week_table
CROSS JOIN (
    SELECT DISTINCT prodname
    FROM dataset
) AS prodname_table
LEFT JOIN dataset ON week_table.solddate = dataset.SOLDDATE
                 AND prodname_table.prodname = dataset.prodname
GROUP BY week_table.solddate, prodname_table.prodname;

select * from dataset00;

-- *첫번째 프로시저 -> pred_prot1 테이블이 존재하지 않으면 생성 (생산예측을 위해 만들어지는 테이블)
DELIMITER //
drop procedure Start0;
CREATE OR REPLACE PROCEDURE Start0()
BEGIN
    IF NOT EXISTS (SELECT * FROM information_schema.tables WHERE table_name = 'pred_prot1') THEN
        CREATE TABLE pred_prot1 AS
        SELECT
            week_table.week AS week,
            prodname_table.prodname AS prodname,
            COALESCE(SUM(ORDER_QUANT), 0) AS order_quant,
            COALESCE(SUM(SOLD_QUANT), 0) AS sold_quant,
            rn
        FROM (
            SELECT DISTINCT DATE_SUB(d.SOLDDATE, INTERVAL (DAYOFWEEK(d.SOLDDATE) + 5) % 7 DAY) AS week,
            DENSE_RANK() OVER (ORDER BY DATE_SUB(d.SOLDDATE, INTERVAL (DAYOFWEEK(d.SOLDDATE) + 5) % 7 DAY)) AS rn
            FROM dataset00 d 
        ) AS week_table
        CROSS JOIN (
            SELECT DISTINCT prodname
            FROM dataset00
        ) AS prodname_table
        LEFT JOIN dataset00 ON week_table.week = DATE_SUB(dataset00.SOLDDATE, INTERVAL (DAYOFWEEK(dataset00.SOLDDATE) + 5) % 7 DAY)
                         AND prodname_table.prodname = dataset00.prodname
        GROUP BY week_table.week, prodname_table.prodname;
        
        ALTER TABLE pred_prot1
        ADD (tem_avg DECIMAL(18, 2),
            hum_avg DECIMAL(18, 2),
            국내건설수주액 INT(33),
            국내기성액 INT(33));
        
        UPDATE pred_prot1
        JOIN (
            SELECT 
                DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY) AS WEEK, 
                ROUND(AVG(D_DAY_2_TEM) OVER (ORDER BY DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY)), 2) AS tem_avg,
                ROUND(AVG(D_DAY_2_hum) OVER (ORDER BY DATE_SUB(SOLDDATE, INTERVAL (DAYOFWEEK(SOLDDATE) + 5) % 7 DAY)), 2) AS hum_avg
            FROM dataset
        ) AS d
        ON pred_prot1.week = d.week
        SET pred_prot1.tem_avg = d.tem_avg,
            pred_prot1.hum_avg = d.hum_avg;
    END IF;
END;
//
DELIMITER ;

-- 임시로 추가로 들어오는 데이터 테이블 생성
drop table new_data;
create table new_data as
select * from dataset00;

UPDATE new_data
SET solddate = DATE_ADD(DATE_ADD(DATE_ADD(solddate, INTERVAL 2 YEAR), INTERVAL 5 MONTH), INTERVAL 17 DAY);

select * from new_data where solddate <= '2021-04-25';

drop table pred_prot1;

select * from pred_prot1 order by week, prodname;

-- *두번째 프로시저 -> pred_prot2가 존재하지 않으면 pred_prot1 테이블을 복사 존재한다면 pred_prot2 에는 없고 pred_prot1 에는 존재하는 행들을 insert 함
-- pred_prot1 테이블에 값이 입력되기 전에 작동하게 할 목적으로 작성

drop table pred_prot2;

DELIMITER //
drop procedure CreateAndCopyTables;
CREATE OR REPLACE PROCEDURE CreateAndCopyTables()
BEGIN
    -- pred_prot2 테이블이 존재하지 않으면 생성
    IF NOT EXISTS (SELECT * FROM information_schema.tables WHERE table_name = 'pred_prot2') THEN
        CREATE TABLE pred_prot2 AS SELECT * FROM pred_prot1 WHERE 0;
    END IF;
   
    -- pred_prot2 테이블에 없는 행 복사
    INSERT INTO pred_prot2
    SELECT p1.* FROM pred_prot1 p1
    LEFT JOIN pred_prot2 p2 ON p1.week = p2.week and p1.prodname = p2.prodname
    WHERE p2.week IS NULL;
	
END;
//
DELIMITER ;

select * from pred_prot2 order by week, prodname;

drop table pred_prot3;

select * from pred_prot3 order by week, prodname;

drop table pred_prot4;

-- *세번째 프로시저 -> pred_prot4가 존재하지 않으면 pred_prot3 테이블을 복사 존재한다면 pred_prot4 에는 없고 pred_prot3 에는 존재하는 행들을 insert 함
-- pred_prot3 는 vscode 에서 코드로 생성 -> 생상예측 우 나온 예측 주별 각 제품 생산량을 추출한 테이블

DELIMITER //
drop procedure CreateAndCopyTables2;
CREATE OR REPLACE PROCEDURE CreateAndCopyTables2()
BEGIN
	-- pred_prot4 테이블이 존재하지 않으면 생성
	IF NOT EXISTS (SELECT * FROM information_schema.tables WHERE table_name = 'pred_prot4') THEN
	    CREATE TABLE pred_prot4 AS SELECT * FROM pred_prot3 WHERE 0;
    
	else
	-- pred_prot4 테이블에 없는 행 복사
	INSERT INTO pred_prot4
	SELECT p1.* FROM pred_prot3 p1
	LEFT JOIN pred_prot4 p2 ON p1.week = p2.week and p1.prodname  = p2.prodname 
	WHERE p2.week IS NULL;
	end if;	
end;
//
DELIMITER ;

select * from pred_prot4;

drop table pred_prot5;

select * from pred_prot5;

-- *네번째 프로시저 pred_prot6가 존재하지 않으면 pred_prot5 테이블을 복사 존재한다면 pred_prot6 에는 없고 pred_prot5 에는 존재하는 행들을 insert 함
-- pred_prot5 

drop table pred_prot6;

DELIMITER //
drop procedure CreateAndCopyTables3;
CREATE OR REPLACE PROCEDURE CreateAndCopyTables3()
BEGIN
	-- pred_prot6 테이블이 존재하지 않으면 생성
	IF NOT EXISTS (SELECT * FROM information_schema.tables WHERE table_name = 'pred_prot6') THEN
	    CREATE TABLE pred_prot6 AS SELECT * FROM pred_prot5 WHERE 0;
    
	else
	-- pred_prot6 테이블에 없는 행 복사
	INSERT INTO pred_prot6
	SELECT p1.* FROM pred_prot5 p1
	LEFT JOIN pred_prot6 p2 ON p1.week = p2.week and p1.사용수 = p2.사용수
	WHERE p2.week IS NULL;
	end if;
END;
//
DELIMITER ;

select * from pred_prot6;

--
select * from pred_prot1 order by week, prodname;
select * from pred_prot3;
select * from pred_prot5;