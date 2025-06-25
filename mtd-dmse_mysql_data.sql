-- MySQL dump 10.13  Distrib 8.0.24, for Linux (x86_64)
--
-- Host: localhost    Database: mtd-dmse
-- ------------------------------------------------------
-- Server version	8.0.24

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_group` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_group`
--

LOCK TABLES `auth_group` WRITE;
/*!40000 ALTER TABLE `auth_group` DISABLE KEYS */;
/*!40000 ALTER TABLE `auth_group` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_group_permissions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `group_id` int NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_group_permissions`
--

LOCK TABLES `auth_group_permissions` WRITE;
/*!40000 ALTER TABLE `auth_group_permissions` DISABLE KEYS */;
/*!40000 ALTER TABLE `auth_group_permissions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_permission` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=90 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_permission`
--

LOCK TABLES `auth_permission` WRITE;
/*!40000 ALTER TABLE `auth_permission` DISABLE KEYS */;
INSERT INTO `auth_permission` VALUES (1,'Can add log entry',1,'add_logentry'),(2,'Can change log entry',1,'change_logentry'),(3,'Can delete log entry',1,'delete_logentry'),(4,'Can view log entry',1,'view_logentry'),(5,'Can add permission',2,'add_permission'),(6,'Can change permission',2,'change_permission'),(7,'Can delete permission',2,'delete_permission'),(8,'Can view permission',2,'view_permission'),(9,'Can add group',3,'add_group'),(10,'Can change group',3,'change_group'),(11,'Can delete group',3,'delete_group'),(12,'Can view group',3,'view_group'),(13,'Can add content type',4,'add_contenttype'),(14,'Can change content type',4,'change_contenttype'),(15,'Can delete content type',4,'delete_contenttype'),(16,'Can view content type',4,'view_contenttype'),(17,'Can add session',5,'add_session'),(18,'Can change session',5,'change_session'),(19,'Can delete session',5,'delete_session'),(20,'Can view session',5,'view_session'),(21,'Can add database_manage',6,'add_database_manage'),(22,'Can change database_manage',6,'change_database_manage'),(23,'Can delete database_manage',6,'delete_database_manage'),(24,'Can view database_manage',6,'view_database_manage'),(25,'Can add database_manage2',7,'add_database_manage2'),(26,'Can change database_manage2',7,'change_database_manage2'),(27,'Can delete database_manage2',7,'delete_database_manage2'),(28,'Can view database_manage2',7,'view_database_manage2'),(29,'Can add 数据集管理',8,'add_datasetmanagement'),(30,'Can change 数据集管理',8,'change_datasetmanagement'),(31,'Can delete 数据集管理',8,'delete_datasetmanagement'),(32,'Can view 数据集管理',8,'view_datasetmanagement'),(33,'Can add early_warning_database',9,'add_early_warning_database'),(34,'Can change early_warning_database',9,'change_early_warning_database'),(35,'Can delete early_warning_database',9,'delete_early_warning_database'),(36,'Can view early_warning_database',9,'view_early_warning_database'),(37,'Can add execute_the_program',10,'add_execute_the_program'),(38,'Can change execute_the_program',10,'change_execute_the_program'),(39,'Can delete execute_the_program',10,'delete_execute_the_program'),(40,'Can view execute_the_program',10,'view_execute_the_program'),(41,'Can add experimental_result',11,'add_experimental_result'),(42,'Can change experimental_result',11,'change_experimental_result'),(43,'Can delete experimental_result',11,'delete_experimental_result'),(44,'Can view experimental_result',11,'view_experimental_result'),(45,'Can add eyi_result',12,'add_eyi_result'),(46,'Can change eyi_result',12,'change_eyi_result'),(47,'Can delete eyi_result',12,'delete_eyi_result'),(48,'Can view eyi_result',12,'view_eyi_result'),(49,'Can add malicious',13,'add_malicious'),(50,'Can change malicious',13,'change_malicious'),(51,'Can delete malicious',13,'delete_malicious'),(52,'Can view malicious',13,'view_malicious'),(53,'Can add malicious_models_manage',14,'add_malicious_models_manage'),(54,'Can change malicious_models_manage',14,'change_malicious_models_manage'),(55,'Can delete malicious_models_manage',14,'delete_malicious_models_manage'),(56,'Can view malicious_models_manage',14,'view_malicious_models_manage'),(57,'Can add malicious_traffic11',15,'add_malicious_traffic11'),(58,'Can change malicious_traffic11',15,'change_malicious_traffic11'),(59,'Can delete malicious_traffic11',15,'delete_malicious_traffic11'),(60,'Can view malicious_traffic11',15,'view_malicious_traffic11'),(61,'Can add model_info',16,'add_model_info'),(62,'Can change model_info',16,'change_model_info'),(63,'Can delete model_info',16,'delete_model_info'),(64,'Can view model_info',16,'view_model_info'),(65,'Can add 模型管理',17,'add_modelmanagement'),(66,'Can change 模型管理',17,'change_modelmanagement'),(67,'Can delete 模型管理',17,'delete_modelmanagement'),(68,'Can view 模型管理',17,'view_modelmanagement'),(69,'Can add models_manage',18,'add_models_manage'),(70,'Can change models_manage',18,'change_models_manage'),(71,'Can delete models_manage',18,'delete_models_manage'),(72,'Can view models_manage',18,'view_models_manage'),(73,'Can add test_dataset',19,'add_test_dataset'),(74,'Can change test_dataset',19,'change_test_dataset'),(75,'Can delete test_dataset',19,'delete_test_dataset'),(76,'Can view test_dataset',19,'view_test_dataset'),(77,'Can add user',20,'add_userinfo'),(78,'Can change user',20,'change_userinfo'),(79,'Can delete user',20,'delete_userinfo'),(80,'Can view user',20,'view_userinfo'),(81,'Can add 检测历史',21,'add_detectionhistory'),(82,'Can change 检测历史',21,'change_detectionhistory'),(83,'Can delete 检测历史',21,'delete_detectionhistory'),(84,'Can view 检测历史',21,'view_detectionhistory'),(85,'Can add data augmentation task',22,'add_dataaugmentationtask'),(86,'Can change data augmentation task',22,'change_dataaugmentationtask'),(87,'Can delete data augmentation task',22,'delete_dataaugmentationtask'),(88,'Can view data augmentation task',22,'view_dataaugmentationtask'),(89,'Access admin page',23,'view');
/*!40000 ALTER TABLE `auth_permission` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `database_info`
--

DROP TABLE IF EXISTS `database_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `database_info` (
  `database_id` int NOT NULL AUTO_INCREMENT,
  `database_name` varchar(20) NOT NULL,
  `database_grouping` varchar(20) NOT NULL,
  `database_instances` varchar(20) NOT NULL,
  `database_features` varchar(20) NOT NULL,
  `create_time` datetime(6) NOT NULL,
  PRIMARY KEY (`database_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `database_info`
--

LOCK TABLES `database_info` WRITE;
/*!40000 ALTER TABLE `database_info` DISABLE KEYS */;
/*!40000 ALTER TABLE `database_info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `database_manage2`
--

DROP TABLE IF EXISTS `database_manage2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `database_manage2` (
  `Database_id` int NOT NULL AUTO_INCREMENT,
  `Database_name` varchar(20) NOT NULL,
  `Database_number` varchar(20) NOT NULL,
  `Database_type` varchar(200) NOT NULL,
  `create_time` datetime(6) NOT NULL,
  PRIMARY KEY (`Database_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `database_manage2`
--

LOCK TABLES `database_manage2` WRITE;
/*!40000 ALTER TABLE `database_manage2` DISABLE KEYS */;
/*!40000 ALTER TABLE `database_manage2` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_admin_log` (
  `id` int NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint unsigned NOT NULL,
  `change_message` longtext NOT NULL,
  `content_type_id` int DEFAULT NULL,
  `user_id` bigint NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6_fk_MTD_userinfo_id` (`user_id`),
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`),
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_MTD_userinfo_id` FOREIGN KEY (`user_id`) REFERENCES `mtd_userinfo` (`id`),
  CONSTRAINT `django_admin_log_chk_1` CHECK ((`action_flag` >= 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_admin_log`
--

LOCK TABLES `django_admin_log` WRITE;
/*!40000 ALTER TABLE `django_admin_log` DISABLE KEYS */;
/*!40000 ALTER TABLE `django_admin_log` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_content_type` (
  `id` int NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_content_type`
--

LOCK TABLES `django_content_type` WRITE;
/*!40000 ALTER TABLE `django_content_type` DISABLE KEYS */;
INSERT INTO `django_content_type` VALUES (1,'admin','logentry'),(3,'auth','group'),(2,'auth','permission'),(4,'contenttypes','contenttype'),(23,'django_rq','queue'),(22,'MTD','dataaugmentationtask'),(6,'MTD','database_manage'),(7,'MTD','database_manage2'),(8,'MTD','datasetmanagement'),(21,'MTD','detectionhistory'),(9,'MTD','early_warning_database'),(10,'MTD','execute_the_program'),(11,'MTD','experimental_result'),(12,'MTD','eyi_result'),(13,'MTD','malicious'),(14,'MTD','malicious_models_manage'),(15,'MTD','malicious_traffic11'),(17,'MTD','modelmanagement'),(18,'MTD','models_manage'),(16,'MTD','model_info'),(19,'MTD','test_dataset'),(20,'MTD','userinfo'),(5,'sessions','session');
/*!40000 ALTER TABLE `django_content_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_migrations` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_migrations`
--

LOCK TABLES `django_migrations` WRITE;
/*!40000 ALTER TABLE `django_migrations` DISABLE KEYS */;
INSERT INTO `django_migrations` VALUES (1,'contenttypes','0001_initial','2025-04-07 14:21:35.046317'),(2,'contenttypes','0002_remove_content_type_name','2025-04-07 14:21:35.071376'),(3,'auth','0001_initial','2025-04-07 14:21:35.159232'),(4,'auth','0002_alter_permission_name_max_length','2025-04-07 14:21:35.188243'),(5,'auth','0003_alter_user_email_max_length','2025-04-07 14:21:35.193232'),(6,'auth','0004_alter_user_username_opts','2025-04-07 14:21:35.197232'),(7,'auth','0005_alter_user_last_login_null','2025-04-07 14:21:35.201241'),(8,'auth','0006_require_contenttypes_0002','2025-04-07 14:21:35.203231'),(9,'auth','0007_alter_validators_add_error_messages','2025-04-07 14:21:35.208232'),(10,'auth','0008_alter_user_username_max_length','2025-04-07 14:21:35.211239'),(11,'auth','0009_alter_user_last_name_max_length','2025-04-07 14:21:35.215240'),(12,'auth','0010_alter_group_name_max_length','2025-04-07 14:21:35.223480'),(13,'auth','0011_update_proxy_permissions','2025-04-07 14:21:35.227479'),(14,'auth','0012_alter_user_first_name_max_length','2025-04-07 14:21:35.231989'),(15,'MTD','0001_initial','2025-04-07 14:21:35.482580'),(16,'admin','0001_initial','2025-04-07 14:21:35.534696'),(17,'admin','0002_logentry_remove_auto_add','2025-04-07 14:21:35.539694'),(18,'admin','0003_logentry_add_action_flag_choices','2025-04-07 14:21:35.544694'),(19,'sessions','0001_initial','2025-04-07 14:21:35.560686'),(20,'MTD','0002_alter_datasetmanagement_category_and_more','2025-04-09 14:56:04.695252'),(21,'MTD','0003_dataaugmentationtask','2025-04-10 09:10:06.254920'),(22,'MTD','0004_alter_dataaugmentationtask_log_file_path','2025-04-10 09:16:43.309554'),(23,'django_rq','0001_initial','2025-04-10 14:20:36.345777'),(24,'MTD','0005_auto_20250411_0925','2025-04-11 01:25:52.037811'),(25,'MTD','0006_auto_20250412_2022','2025-04-12 12:22:10.633648');
/*!40000 ALTER TABLE `django_migrations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_session`
--

LOCK TABLES `django_session` WRITE;
/*!40000 ALTER TABLE `django_session` DISABLE KEYS */;
INSERT INTO `django_session` VALUES ('27qo4umbpg3phgs27nzi47q98vxejvrp','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uQQQx:sLB4n8MZjbCBnNKGTOf_ZnEyA3LzJ_7SMiZtrN5-Csg','2025-06-28 12:56:31.835745'),('83ghz2kirzquntc1ai8li46ggyap1znq','.eJxVjEEOwiAQAP_C2RB22QJ69O4bCCxUqgaS0p6MfzckPeh1ZjJv4cO-Fb_3vPoliYsAcfplMfAz1yHSI9R7k9zqti5RjkQetstbS_l1Pdq_QQm9jC1qUmdkl7LWju3MFNFGjZNNrGciYycgUM6gBjDIymRnLeQE0SAp8fkCspc2Qw:1u3Bz3:3a35cvHdZUZRemxpbIau0IeOU6gW6npPtEEni_oE9Sc','2025-04-25 10:51:41.705463'),('8r81v1ut2r1bhtbb1z2ie1ou5nydf7jc','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uMpVk:sk-xZajHvjlJ4ykHkJRcZcqgebulDtcD7IPdLT65KQw','2025-06-18 14:54:36.515724'),('brrhwtcjjpww0tgccsx96jo6bk2bel66','.eJyNjr1SwzAQhN9FdeQ5yfpNSWo6qD2nu3NiIDJjORXDu-PMpAgMRdrd_Xb3Sw14WU_DpckyTKz2yqjdvVaQ3qVeDX7Depw7muu6TKW7Rrqb27rnmeXj6Zb9VXDCdtpoSWOBMXjhnkJAY6xhY3w0TEBgSx5HJFss4SYXQfAIESyl6H1vQt5KP5f5uEhrQ5bowMVRWybRLjJpxATap2CBepPZJ7U3ADvFuGKTdah4lgc5dXh51Yd5quepynK_m1zJjL3TYH3UjiHo7CxqSs667a_PEP_bfYz7s_v9A5ywe6Q:1u5Pot:yL332bqwhFmM0cuBkfT86FYJkSg3hnxH0g014J-REKQ','2025-05-01 14:02:23.254492'),('bxj9074hloa4ldwtj8ap8khqx6ond9o5','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uSYxO:3e5yygQiAzMVLDQfB6cUbQlbSXDl46zdZ8HSsv7n7pM','2025-07-04 10:26:50.668181'),('bzir90vsf8s88t5mgk7l1i6qmp73206t','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uNpXS:Ut7428KGvJvHtGoy4drkLw4MublhFAY4pkfN4qcftWc','2025-06-21 09:08:30.928309'),('cc5pxzqeye155dgf8865j2pn2wmc7y8t','.eJyNlE1v2zAMhv-Lz2UgUdRXb2t7XIFhbc8GRUmNt9Yp7ORU9L9PwbItwxzMN0Hiw_elSOm96_mw3_aHuUz9kLvrTndX53uJ5XsZjwf5G4_Pu43sxv00pM0xZHM6nTf3u1xebk6xfyXY8rw9pkVDKqKEXIwJ4qtQQp8MWp_FVCLnrSatgkOjtUNRrgTvdck6OSTVksphmsq47zPveS77fuTX0jLfPj7B7W4YX4exTGdhb9PueSrz3F1rpa66_cTDOIzP_RtP_Np237v_JHqZjr4LUFuPu2Eu_Szbkg8vx2DZzS2s-7jqfun0xK0eLwGccQ6IjIfEWAGNNyFYJAzxZOY3wylpwWigltqYTAxJ2abJURdC5arHE3PudiXXPT083sLDdqiH7ky0Brbso4OqjkZVbEZdyKAVSaYchQwtia7jLogqUaSKJ0giGchXBiZnIceoNJHPmdOS6Drun-b96YpoVckR6OI9kG2mA1IEa21CSVbaMC7pruMuFEsVdZtugsylmQ41QMKqQKNgm3tN1i22dR13udjKjJnFA-efw9BWjkurQWHxUSpms9jZVdyFYlPWtWabwAcrQKUWYEMekK0SJ4VFFsdpHXdBNLCjyNLG3mvVHBcEZouga_tjMKGLui6JruMu37COlEJNBQxq295AksYbgaIpp5B95GSXdNdx3c3w-eHx_lwQvUs5mwQmt-dOPllIOlTwnJTGhhW9KLiO6-7uvsDdp6_dxw_JB82n:1u3C1Y:BnUQIs3-nZ_VOpPG90x8XumCcK_omdsYfgOfGULSXj8','2025-04-25 10:54:16.826573'),('f6q82iahifwx65uno4eds6q49m5wt9eo','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uMpSs:v8qf4NrgWQnTJnJ6920--L2b4qrMZ-f4fTxPHPkjgMc','2025-06-18 14:51:38.971008'),('g9gugep9dibud7kkygw0tuxwnf0klixx','.eJxVjMsOwiAQRf-FtSE8CgMu3fsNhBmmUjU0Ke3K-O_apAvd3nPOfYmUt7WmrfOSpiLOwovT74aZHtx2UO653WZJc1uXCeWuyIN2eZ0LPy-H-3dQc6_fOjpSFjzamP2I2jH5gbUPSIDRGxWZNIGj4jizCZbBKA3E0boyBhjE-wPlyzfs:1uN7bV:N4DiubzedsFO3SPad1cSqX6mi9Rbid_yA0zq03hhVyY','2025-06-19 10:13:45.396751'),('jojqtsg0p2309vgyi7vrazg50jolqhe9','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uMpRD:34OOThINuOJl0y_ngikLlKPMVYRFjIAzYSzgV8wuczk','2025-06-18 14:49:55.832677'),('troa9t2xq6sli9p0zvuxfylf3oof5ro3','.eJyNjDsOwjAQBe_imkT-rOM1JdRwBWttrwm_BMWhQtydIFFAR_tm5j1EoPvch3vlKRyzWAsrVt9bpHTm4Q3yiYbD2KZxmKdjbN9K-6G13Y2ZL5uP-3PQU-2XulhMrqBzGXSUrJG1YauYnWXlsRA4w95qD9o6UzoN0QBiAozZKsfL6W0aDxPXGmS3RKyg6dD7BtBS4xOYptOFJXDO0UmxVlKuRKaZKs9hoCv_2Yntfi-eL0oIVbA:1uN6Qj:pOGjNWSMRyknM8e6eTx99HOak7CfiuK9z9YG7eFjJs4','2025-06-19 08:58:33.731049'),('v935tpqj2i26df6r6ucb8hapqn42jqwp','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uQHAG:Gmgy06fjnlOj_RH93XhBUvaiCQpks7-bfZ_DmWSxZJI','2025-06-28 03:02:40.792179'),('xm3uxhbed52753e8jmrawwlzar3dw9bg','.eJxVjDsOwjAQBe_iGln4s-yakp4zWP6scQDZUpxUiLtDpBTQvpl5L-HDulS_Dp79lMVZgDj8bjGkB7cN5Htoty5Tb8s8RbkpcqdDXnvm52V3_w5qGPVbF6CEhRCz1fHImlgbBsWMwMpRCRYNO9DOakBTTtpGY4mSpZhBIYv3B9j0N0E:1uQme4:YzCkWR040XrnoNNRVt6YSRYrbMFB6O_O2wOulEUDyi8','2025-06-29 12:39:32.156600');
/*!40000 ALTER TABLE `django_session` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `early_warning_database`
--

DROP TABLE IF EXISTS `early_warning_database`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `early_warning_database` (
  `id` int NOT NULL AUTO_INCREMENT,
  `tool_name` varchar(40) NOT NULL,
  `testcase_name` varchar(40) NOT NULL,
  `indicator_a` varchar(40) NOT NULL,
  `indicator_p` varchar(40) NOT NULL,
  `indicator_r` varchar(40) NOT NULL,
  `indicator_f` varchar(40) NOT NULL,
  `statu` varchar(40) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `early_warning_database`
--

LOCK TABLES `early_warning_database` WRITE;
/*!40000 ALTER TABLE `early_warning_database` DISABLE KEYS */;
/*!40000 ALTER TABLE `early_warning_database` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `execute_the_program`
--

DROP TABLE IF EXISTS `execute_the_program`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `execute_the_program` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(40) NOT NULL,
  `path` varchar(200) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `execute_the_program`
--

LOCK TABLES `execute_the_program` WRITE;
/*!40000 ALTER TABLE `execute_the_program` DISABLE KEYS */;
/*!40000 ALTER TABLE `execute_the_program` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `experimental_result`
--

DROP TABLE IF EXISTS `experimental_result`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `experimental_result` (
  `id` int NOT NULL AUTO_INCREMENT,
  `tool_name` varchar(40) NOT NULL,
  `testcase_name` varchar(40) NOT NULL,
  `indicator_a` varchar(40) NOT NULL,
  `indicator_p` varchar(40) NOT NULL,
  `indicator_r` varchar(40) NOT NULL,
  `indicator_f` varchar(40) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `experimental_result`
--

LOCK TABLES `experimental_result` WRITE;
/*!40000 ALTER TABLE `experimental_result` DISABLE KEYS */;
/*!40000 ALTER TABLE `experimental_result` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `eyi_result`
--

DROP TABLE IF EXISTS `eyi_result`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `eyi_result` (
  `id` int NOT NULL AUTO_INCREMENT,
  `models_name` varchar(40) NOT NULL,
  `database_name` varchar(40) NOT NULL,
  `average_a` varchar(40) NOT NULL,
  `average_p` varchar(40) NOT NULL,
  `average_r` varchar(40) NOT NULL,
  `average_f` varchar(40) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `eyi_result`
--

LOCK TABLES `eyi_result` WRITE;
/*!40000 ALTER TABLE `eyi_result` DISABLE KEYS */;
/*!40000 ALTER TABLE `eyi_result` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `malicious`
--

DROP TABLE IF EXISTS `malicious`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `malicious` (
  `Mid` int NOT NULL AUTO_INCREMENT,
  `Mname` varchar(200) NOT NULL,
  `Mpath` varchar(200) NOT NULL,
  PRIMARY KEY (`Mid`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `malicious`
--

LOCK TABLES `malicious` WRITE;
/*!40000 ALTER TABLE `malicious` DISABLE KEYS */;
INSERT INTO `malicious` VALUES (1,'CWE','TEST');
/*!40000 ALTER TABLE `malicious` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `malicious_models_manage`
--

DROP TABLE IF EXISTS `malicious_models_manage`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `malicious_models_manage` (
  `Malicious_model_id` int NOT NULL AUTO_INCREMENT,
  `Malicious_model_name` varchar(20) DEFAULT NULL,
  `Malicious_model_grouping` varchar(20) DEFAULT NULL,
  `create_time` datetime(6) DEFAULT NULL,
  `is_Bidirectional` varchar(20) DEFAULT NULL,
  `is_feature` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`Malicious_model_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `malicious_models_manage`
--

LOCK TABLES `malicious_models_manage` WRITE;
/*!40000 ALTER TABLE `malicious_models_manage` DISABLE KEYS */;
/*!40000 ALTER TABLE `malicious_models_manage` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `model_info`
--

DROP TABLE IF EXISTS `model_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `model_info` (
  `model_id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(200) NOT NULL,
  `model_info_url` varchar(200) NOT NULL,
  `upload_date` datetime(6) NOT NULL,
  PRIMARY KEY (`model_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `model_info`
--

LOCK TABLES `model_info` WRITE;
/*!40000 ALTER TABLE `model_info` DISABLE KEYS */;
/*!40000 ALTER TABLE `model_info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `models_info`
--

DROP TABLE IF EXISTS `models_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `models_info` (
  `model_id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(20) DEFAULT NULL,
  `model_grouping` varchar(20) DEFAULT NULL,
  `create_time` datetime(6) DEFAULT NULL,
  `is_incremental_learning` varchar(20) DEFAULT NULL,
  `is_multiple` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`model_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `models_info`
--

LOCK TABLES `models_info` WRITE;
/*!40000 ALTER TABLE `models_info` DISABLE KEYS */;
/*!40000 ALTER TABLE `models_info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_datasetmanagement`
--

DROP TABLE IF EXISTS `mtd_datasetmanagement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_datasetmanagement` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `category` varchar(10) NOT NULL,
  `upload_time` datetime(6) NOT NULL,
  `data_file` varchar(100) NOT NULL,
  `size` int unsigned NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`),
  CONSTRAINT `mtd_datasetmanagement_chk_1` CHECK ((`size` >= 0))
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_datasetmanagement`
--

LOCK TABLES `mtd_datasetmanagement` WRITE;
/*!40000 ALTER TABLE `mtd_datasetmanagement` DISABLE KEYS */;
INSERT INTO `mtd_datasetmanagement` VALUES (2,'USTC-TFC2016','RGB','2024-12-09 12:14:37.892462','datasets/USTC-TFC2016-master.zip',390500),(3,'CTU','RGB','2025-01-09 12:15:18.367627','datasets/CTU.zip',143798),(4,'ISAC','RGB','2025-01-09 12:15:45.015408','datasets/ISAC.zip',229357),(5,'DDoSdata','CSV','2025-02-09 12:20:16.738297','datasets/DDoSdata.csv',211520),(9,'KDD-CUP','CSV','2025-03-11 01:55:31.139401','datasets/KDDTest.csv',561468),(10,'ISAC-PCAP','PCAP','2024-11-12 03:03:09.655600','datasets/ISAC_WrJz68h.zip',163432),(11,'CTU-PACP','PCAP','2025-02-11 05:23:47.226775','datasets/CTU_3R2vNnn.zip',293841),(14,'USTC-Enhanced ','RGB','2025-02-15 13:54:47.072663','datasets/USTC.zip',352814),(15,'CTU-Coinminer','RGB','2024-10-15 04:28:20.264378','datasets/CTU-Coinminer.zip',5600),(16,'USTC-Shifu','RGB','2024-09-17 04:28:48.202224','datasets/USTC-Shifu.zip',7256);
/*!40000 ALTER TABLE `mtd_datasetmanagement` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_detectionhistory`
--

DROP TABLE IF EXISTS `mtd_detectionhistory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_detectionhistory` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `detection_time` datetime(6) NOT NULL,
  `is_malicious` tinyint(1) NOT NULL,
  `accuracy` decimal(5,2) DEFAULT NULL,
  `report` varchar(100) DEFAULT NULL,
  `dataset_id` bigint NOT NULL,
  `model_id` bigint NOT NULL,
  `F1_score` decimal(5,2) DEFAULT NULL,
  `FPR` decimal(5,2) DEFAULT NULL,
  `TPR` decimal(5,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `MTD_detectionhistory_dataset_id_18c9707c_fk_MTD_datas` (`dataset_id`),
  KEY `MTD_detectionhistory_model_id_fe299141_fk_MTD_modelmanagement_id` (`model_id`),
  CONSTRAINT `MTD_detectionhistory_dataset_id_18c9707c_fk_MTD_datas` FOREIGN KEY (`dataset_id`) REFERENCES `mtd_datasetmanagement` (`id`),
  CONSTRAINT `MTD_detectionhistory_model_id_fe299141_fk_MTD_modelmanagement_id` FOREIGN KEY (`model_id`) REFERENCES `mtd_modelmanagement` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=71 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_detectionhistory`
--

LOCK TABLES `mtd_detectionhistory` WRITE;
/*!40000 ALTER TABLE `mtd_detectionhistory` DISABLE KEYS */;
INSERT INTO `mtd_detectionhistory` VALUES (23,'2025-03-02 10:59:37.950570',0,94.44,'',2,1,93.96,0.69,93.49),(24,'2025-02-11 04:59:48.881139',0,93.91,'',10,1,93.25,0.53,92.60),(25,'2025-01-05 09:00:52.594604',0,86.27,'',2,10,86.15,3.62,86.04),(26,'2025-01-02 08:09:01.743740',0,92.10,'',2,9,91.42,0.87,90.74),(27,'2025-02-13 12:16:29.857275',1,99.21,'',2,8,99.20,0.13,98.35),(28,'2024-12-20 12:02:29.303714',1,96.82,'',2,6,96.78,0.41,96.74),(29,'2024-12-11 10:02:37.964584',1,96.01,'',2,5,95.88,0.46,95.74),(30,'2024-11-10 09:02:51.135589',0,86.53,'',10,2,86.15,3.18,85.77),(31,'2024-11-09 06:02:59.002816',0,88.12,'',2,2,87.70,2.66,87.29),(32,'2024-11-03 11:06:13.812521',1,98.17,'',2,11,97.77,0.31,97.37),(33,'2025-01-28 08:10:41.589261',1,90.96,'',9,9,90.11,0.76,89.28),(35,'2024-10-11 07:25:43.436147',0,95.16,'',3,19,94.79,0.99,94.42),(36,'2025-02-07 12:14:17.398486',0,99.45,'',3,8,98.64,0.12,97.85),(37,'2024-09-28 07:28:59.854612',1,97.11,'',3,16,96.95,0.16,96.79),(38,'2024-09-27 07:29:12.872724',0,94.06,'',3,15,94.04,0.95,94.03),(39,'2024-09-26 11:29:25.564217',0,95.82,'',3,14,94.88,0.41,93.95),(40,'2024-09-26 07:29:43.189350',0,96.81,'',3,13,96.26,0.28,95.72),(41,'2024-09-26 07:30:34.311106',0,95.76,'',3,19,95.36,1.14,94.96),(42,'2024-09-26 07:31:03.022106',1,96.75,'',3,12,96.22,0.46,95.69),(43,'2024-09-25 10:43:29.923333',1,96.63,'',11,16,96.00,0.14,95.38),(44,'2024-09-25 10:44:34.267694',1,93.32,'',11,19,92.61,0.96,91.91),(45,'2024-09-25 10:45:41.084756',0,92.56,'',11,19,92.43,1.26,92.30),(46,'2024-09-24 10:46:17.049744',1,93.58,'',11,19,93.24,1.33,92.90),(47,'2024-09-24 10:52:30.485279',0,95.46,'',4,19,94.72,1.31,93.99),(49,'2024-09-24 02:53:32.675788',1,91.12,'',10,15,90.86,1.02,90.60),(50,'2024-09-24 02:54:08.471460',0,91.44,'',10,15,91.11,0.94,90.78),(51,'2024-09-24 11:14:43.139422',0,95.58,'',4,19,94.58,1.30,93.61),(52,'2024-09-23 11:14:51.240811',0,96.36,'',4,18,95.60,0.59,94.86),(53,'2025-04-12 11:15:00.788253',0,98.23,'',4,16,97.71,0.17,97.20),(54,'2025-04-12 12:54:40.341136',0,92.06,'',11,15,91.20,1.23,90.35),(55,'2025-02-23 14:01:57.504589',0,99.40,'',5,8,98.45,0.14,97.52),(56,'2024-09-23 11:21:07.551615',0,97.39,'',9,13,97.21,0.39,97.04),(57,'2024-09-23 13:22:19.177148',1,92.33,'',2,9,92.26,0.73,92.20),(58,'2025-01-23 13:40:58.198799',0,99.45,'',4,8,99.12,0.12,98.98),(59,'2024-09-23 13:41:10.550041',0,98.19,'',4,11,97.82,0.28,97.44),(60,'2025-01-23 13:55:25.107400',1,99.99,'',14,8,99.99,0.09,99.38),(61,'2025-01-25 13:55:36.401695',0,91.37,'',14,9,90.36,0.95,89.38),(62,'2024-09-22 13:55:49.915959',1,94.31,'',14,4,93.83,0.70,93.36),(63,'2024-09-22 13:55:59.366262',1,96.00,'',14,5,95.14,0.44,94.30),(64,'2024-09-22 13:56:24.081134',1,93.42,'',5,1,93.02,0.57,92.62),(65,'2024-09-22 13:56:34.907356',0,97.91,'',5,11,97.19,0.26,96.48),(66,'2025-01-22 13:57:57.288754',1,97.57,'',11,8,97.37,0.14,97.17),(67,'2025-02-23 08:58:06.448245',1,96.91,'',10,8,96.73,0.12,96.56),(68,'2025-02-22 16:58:24.031444',1,99.32,'',5,8,98.74,0.12,98.17),(69,'2025-02-21 13:58:32.532552',1,99.65,'',9,8,98.77,0.14,97.91),(70,'2025-03-20 10:59:54.816923',1,96.97,'',3,13,96.39,0.28,95.82);
/*!40000 ALTER TABLE `mtd_detectionhistory` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_malicious_traffic11`
--

DROP TABLE IF EXISTS `mtd_malicious_traffic11`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_malicious_traffic11` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(200) NOT NULL,
  `path` varchar(200) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_malicious_traffic11`
--

LOCK TABLES `mtd_malicious_traffic11` WRITE;
/*!40000 ALTER TABLE `mtd_malicious_traffic11` DISABLE KEYS */;
/*!40000 ALTER TABLE `mtd_malicious_traffic11` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_modelmanagement`
--

DROP TABLE IF EXISTS `mtd_modelmanagement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_modelmanagement` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `category` varchar(100) NOT NULL,
  `upload_time` datetime(6) NOT NULL,
  `model_file` varchar(100) NOT NULL,
  `description` longtext,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=23 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_modelmanagement`
--

LOCK TABLES `mtd_modelmanagement` WRITE;
/*!40000 ALTER TABLE `mtd_modelmanagement` DISABLE KEYS */;
INSERT INTO `mtd_modelmanagement` VALUES (1,'CNN','CNN','2024-09-09 21:54:48.000000','models/CNN.pt','CNN主要用于处理具有网格结构的数据,如图像；使用卷积层提取特征,池化层降维，擅长捕捉局部特征和空间关系，在图像识别、目标检测等任务中表现出色，但容易过拟合'),(2,'RNN','RNN','2024-09-10 14:00:58.148767','models/RNN__final_0.pth','RNN循环神经网络设计用于处理序列数据\r\n有内部状态(记忆),可以记住之前的信息，适合处理时间序列数据,如自然语言处理，存在长期依赖问题'),(4,'LSTM','LSTM','2024-09-11 14:17:51.030556','models/LSTM_model_best_CTU_0.pt','LSTM是RNN的一种变体,解决了长期依赖问题，有门控机制:输入门、遗忘门、输出门，能够长期保存重要信息,忘记不相关信息，在语音识别、机器翻译等任务中广泛应用'),(5,'TCN','TCN','2024-09-20 14:19:02.919772','models/meta_TCN_model.pt','TCN模型结合了CNN和RNN的优点，使用膨胀卷积,可以捕捉长期依赖，并行计算效率高,训练速度较快，在时间序列预测、音频生成等任务中表现良好'),(6,'BiLSTM','BiLSTM','2024-09-28 14:23:33.053559','models/BiLSTM_best_USTC.pt','LSTM的扩展,包含两个方向的LSTM，可以同时考虑过去和未来的上下文信息，在许多序列标注任务中表现优秀'),(8,'DMSE (Deep-Multi-Stacking-Ensemble)','DMSE','2025-01-28 14:49:48.822606','models/Meta_BiTCN_best_USTC_1.pth','使用CNN、LSTM、TCN、BiLSTM、BiTCN作为基学习器，从时序和空间卷积两方面充分学习网络流量的特征，使用多堆叠集成作为集成学习的集成策略，极大的增强检测模型的泛化能力和稳定性，使用BiTCN作为二层堆叠集成的元学习器，可以学习到流量的高维时序和空间特征，增强对流量的识别效果和精确度。'),(9,'ResNet','Deep_Learning','2024-10-09 14:56:14.012121','models/Resnet34_CTU_0_X2FV4ru.pth','ResNet（Residual Network）是一种深度卷积神经网络架构，旨在解决深度学习中的梯度消失和模型训练困难等问题。ResNet 的核心创新是在网络中引入了“残差连接”（skip connections），使得信息能够在网络中直接流动。'),(10,'EfficientNet','Deep_Learning','2024-10-15 14:57:34.473932','models/Efficientnet_model_final_USTC_0.pth','EfficientNet 是一种高效的卷积神经网络架构，首次由 Google 研究团队于 2019 年提出。其主要目标是通过优化网络的深度（Depth）、宽度（Width）和分辨率（Resolution）来实现更高的效率与准确率，从而取得最佳的模型性能。'),(11,'BiTCN','BiTCN','2024-10-16 05:04:54.761522','models/BiTCN_best_USTC_8.pt','双向时序卷积网络'),(12,'Meta-CNN','CNN','2024-11-12 05:32:33.285917','models/meta_CNN_best_CTU_4.pth','CNN作为stacking堆叠集成的第二层元学习器的模型'),(13,'Meta-BiLSTM','BiLSTM','2024-11-25 05:33:29.699797','models/meta_BiLSTM_best_6.pth','BiLSTM模型作为stacking堆叠集成的第二层元学习器的模型'),(14,'Meta-LSTM','LSTM','2024-12-12 05:34:01.901087','models/meta_LSTM_best_5.pt','LSTM模型作为stacking堆叠集成的第二层元学习器的模型'),(15,'Meta-EfficientNet','Machine_Learning','2025-01-12 05:35:32.238677','models/meta_Efficientnet_model_best_CTU_5.pth','EfficientNet模型作为stacking堆叠集成的第二层元学习器的模型'),(16,'Meta-TCN','TCN','2025-01-12 05:36:14.935361','models/Meta_TCN_best_CTU_4.pth','TCN模型作为stacking堆叠集成的第二层元学习器的模型'),(18,'Meta-ResNet','Deep_Learning','2025-02-15 06:24:30.786689','models/meta_ResNet_model_best_9.pth','ResNet模型作为stacking堆叠集成的第二层元学习器的模型'),(19,'Meta-RNN','RNN','2025-02-16 06:24:51.638662','models/meta_RNN_CTU_best_9.pth','RNN模型作为stacking堆叠集成的第二层元学习器的模型');
/*!40000 ALTER TABLE `mtd_modelmanagement` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_userinfo`
--

DROP TABLE IF EXISTS `mtd_userinfo`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_userinfo` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  `phone` varchar(11) NOT NULL,
  `sex` varchar(1) NOT NULL,
  `birth` date DEFAULT NULL,
  `avatar` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_userinfo`
--

LOCK TABLES `mtd_userinfo` WRITE;
/*!40000 ALTER TABLE `mtd_userinfo` DISABLE KEYS */;
INSERT INTO `mtd_userinfo` VALUES (1,'pbkdf2_sha256$260000$sbspRhBjjw7Z0n2CDq8Jfo$bXlL4tKb8xoE+DNcnlpSbsH0NGAMDxfXkpjOQReufQA=','2025-04-17 14:01:55.662101',0,'Solitude_zy','Yang','z','q2205773452@163.com',0,1,'2025-04-07 14:22:48.726813','19599965629','M','2003-01-19','avatars/猫猫1.jpeg'),(2,'pbkdf2_sha256$260000$1NBCe7vNro8m4wqjXHQHcl$qmG/C8W9TM9jWc6lnwCJf3qqHsMTD5KNjHoHAQ1IuD8=','2025-04-12 13:12:22.716247',0,'zy','','','q2205773452@163.com',0,1,'2025-04-12 13:12:22.598386','','',NULL,''),(3,'pbkdf2_sha256$260000$dGXRsyQ5QER37Qb99DYuKv$thGpHoSwLTuihFTWZpFKCNZrMClC8N1Da4zcHaOC4O0=','2025-04-12 13:13:01.607169',0,'WYY','','','xztszy@gmail.com',0,1,'2025-04-12 13:13:01.495981','','',NULL,''),(4,'pbkdf2_sha256$260000$uQXa6PYCpLBjt5EemO3X0g$ewt0RCtNc29trI28A7ns08pF+UyXzFhMPt5xM7UbvdU=','2025-04-13 12:54:34.124722',0,'wy','','','xztszy@gmail.com',0,1,'2025-04-12 13:17:37.504705','','',NULL,'avatars/猫猫2.jpeg'),(5,'pbkdf2_sha256$260000$Qph35ETbnfLZktCLMC1ca9$NOucsoR66lzN9TKEVAntw4V/2QN9Fl86DCHXnBGmt4I=','2025-06-20 10:26:50.651012',0,'admin','','','q2205773452@163.com',0,1,'2025-06-04 14:41:10.830082','','',NULL,'avatars/doge.jpg'),(6,'pbkdf2_sha256$260000$8UsH1VQbpsyWxiwLJPF65T$1estwsOPlwPQwwHDtlYA36q1OfJw8O55MfRqqHXWYOc=','2025-06-05 10:13:45.389075',0,'awan','','','3056571032@qq.com',0,1,'2025-06-05 10:13:45.220803','','',NULL,'avatars/下载.jpg');
/*!40000 ALTER TABLE `mtd_userinfo` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_userinfo_groups`
--

DROP TABLE IF EXISTS `mtd_userinfo_groups`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_userinfo_groups` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userinfo_id` bigint NOT NULL,
  `group_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `MTD_userinfo_groups_userinfo_id_group_id_849b9305_uniq` (`userinfo_id`,`group_id`),
  KEY `MTD_userinfo_groups_group_id_4b76e195_fk_auth_group_id` (`group_id`),
  CONSTRAINT `MTD_userinfo_groups_group_id_4b76e195_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`),
  CONSTRAINT `MTD_userinfo_groups_userinfo_id_b3d45e72_fk_MTD_userinfo_id` FOREIGN KEY (`userinfo_id`) REFERENCES `mtd_userinfo` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_userinfo_groups`
--

LOCK TABLES `mtd_userinfo_groups` WRITE;
/*!40000 ALTER TABLE `mtd_userinfo_groups` DISABLE KEYS */;
/*!40000 ALTER TABLE `mtd_userinfo_groups` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `mtd_userinfo_user_permissions`
--

DROP TABLE IF EXISTS `mtd_userinfo_user_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mtd_userinfo_user_permissions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userinfo_id` bigint NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `MTD_userinfo_user_permis_userinfo_id_permission_i_8fdc5d9d_uniq` (`userinfo_id`,`permission_id`),
  KEY `MTD_userinfo_user_pe_permission_id_9b8fedac_fk_auth_perm` (`permission_id`),
  CONSTRAINT `MTD_userinfo_user_pe_permission_id_9b8fedac_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `MTD_userinfo_user_pe_userinfo_id_d7fc6a7c_fk_MTD_useri` FOREIGN KEY (`userinfo_id`) REFERENCES `mtd_userinfo` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mtd_userinfo_user_permissions`
--

LOCK TABLES `mtd_userinfo_user_permissions` WRITE;
/*!40000 ALTER TABLE `mtd_userinfo_user_permissions` DISABLE KEYS */;
/*!40000 ALTER TABLE `mtd_userinfo_user_permissions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `test_dataset`
--

DROP TABLE IF EXISTS `test_dataset`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `test_dataset` (
  `test_id` int NOT NULL AUTO_INCREMENT,
  `test_name` varchar(40) NOT NULL,
  `test_path` varchar(200) NOT NULL,
  `upload_date` datetime(6) NOT NULL,
  PRIMARY KEY (`test_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `test_dataset`
--

LOCK TABLES `test_dataset` WRITE;
/*!40000 ALTER TABLE `test_dataset` DISABLE KEYS */;
/*!40000 ALTER TABLE `test_dataset` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping events for database 'mtd-dmse'
--

--
-- Dumping routines for database 'mtd-dmse'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-06-21 10:37:37
