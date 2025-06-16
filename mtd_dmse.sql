/*
 Navicat Premium Data Transfer

 Source Server         : link_by_zy
 Source Server Type    : MySQL
 Source Server Version : 80032 (8.0.32)
 Source Host           : localhost:3306
 Source Schema         : mtd_dmse

 Target Server Type    : MySQL
 Target Server Version : 80032 (8.0.32)
 File Encoding         : 65001

 Date: 13/06/2025 18:56:11
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for auth_group
-- ----------------------------
DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE `auth_group`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_group
-- ----------------------------

-- ----------------------------
-- Table structure for auth_group_permissions
-- ----------------------------
DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE `auth_group_permissions`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `group_id` int NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `auth_group_permissions_group_id_permission_id_0cd325b0_uniq`(`group_id` ASC, `permission_id` ASC) USING BTREE,
  INDEX `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm`(`permission_id` ASC) USING BTREE,
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_group_permissions
-- ----------------------------

-- ----------------------------
-- Table structure for auth_permission
-- ----------------------------
DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE `auth_permission`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `content_type_id` int NOT NULL,
  `codename` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `auth_permission_content_type_id_codename_01ab375a_uniq`(`content_type_id` ASC, `codename` ASC) USING BTREE,
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 90 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_permission
-- ----------------------------
INSERT INTO `auth_permission` VALUES (1, 'Can add log entry', 1, 'add_logentry');
INSERT INTO `auth_permission` VALUES (2, 'Can change log entry', 1, 'change_logentry');
INSERT INTO `auth_permission` VALUES (3, 'Can delete log entry', 1, 'delete_logentry');
INSERT INTO `auth_permission` VALUES (4, 'Can view log entry', 1, 'view_logentry');
INSERT INTO `auth_permission` VALUES (5, 'Can add permission', 2, 'add_permission');
INSERT INTO `auth_permission` VALUES (6, 'Can change permission', 2, 'change_permission');
INSERT INTO `auth_permission` VALUES (7, 'Can delete permission', 2, 'delete_permission');
INSERT INTO `auth_permission` VALUES (8, 'Can view permission', 2, 'view_permission');
INSERT INTO `auth_permission` VALUES (9, 'Can add group', 3, 'add_group');
INSERT INTO `auth_permission` VALUES (10, 'Can change group', 3, 'change_group');
INSERT INTO `auth_permission` VALUES (11, 'Can delete group', 3, 'delete_group');
INSERT INTO `auth_permission` VALUES (12, 'Can view group', 3, 'view_group');
INSERT INTO `auth_permission` VALUES (13, 'Can add content type', 4, 'add_contenttype');
INSERT INTO `auth_permission` VALUES (14, 'Can change content type', 4, 'change_contenttype');
INSERT INTO `auth_permission` VALUES (15, 'Can delete content type', 4, 'delete_contenttype');
INSERT INTO `auth_permission` VALUES (16, 'Can view content type', 4, 'view_contenttype');
INSERT INTO `auth_permission` VALUES (17, 'Can add session', 5, 'add_session');
INSERT INTO `auth_permission` VALUES (18, 'Can change session', 5, 'change_session');
INSERT INTO `auth_permission` VALUES (19, 'Can delete session', 5, 'delete_session');
INSERT INTO `auth_permission` VALUES (20, 'Can view session', 5, 'view_session');
INSERT INTO `auth_permission` VALUES (21, 'Can add database_manage', 6, 'add_database_manage');
INSERT INTO `auth_permission` VALUES (22, 'Can change database_manage', 6, 'change_database_manage');
INSERT INTO `auth_permission` VALUES (23, 'Can delete database_manage', 6, 'delete_database_manage');
INSERT INTO `auth_permission` VALUES (24, 'Can view database_manage', 6, 'view_database_manage');
INSERT INTO `auth_permission` VALUES (25, 'Can add database_manage2', 7, 'add_database_manage2');
INSERT INTO `auth_permission` VALUES (26, 'Can change database_manage2', 7, 'change_database_manage2');
INSERT INTO `auth_permission` VALUES (27, 'Can delete database_manage2', 7, 'delete_database_manage2');
INSERT INTO `auth_permission` VALUES (28, 'Can view database_manage2', 7, 'view_database_manage2');
INSERT INTO `auth_permission` VALUES (29, 'Can add 数据集管理', 8, 'add_datasetmanagement');
INSERT INTO `auth_permission` VALUES (30, 'Can change 数据集管理', 8, 'change_datasetmanagement');
INSERT INTO `auth_permission` VALUES (31, 'Can delete 数据集管理', 8, 'delete_datasetmanagement');
INSERT INTO `auth_permission` VALUES (32, 'Can view 数据集管理', 8, 'view_datasetmanagement');
INSERT INTO `auth_permission` VALUES (33, 'Can add early_warning_database', 9, 'add_early_warning_database');
INSERT INTO `auth_permission` VALUES (34, 'Can change early_warning_database', 9, 'change_early_warning_database');
INSERT INTO `auth_permission` VALUES (35, 'Can delete early_warning_database', 9, 'delete_early_warning_database');
INSERT INTO `auth_permission` VALUES (36, 'Can view early_warning_database', 9, 'view_early_warning_database');
INSERT INTO `auth_permission` VALUES (37, 'Can add execute_the_program', 10, 'add_execute_the_program');
INSERT INTO `auth_permission` VALUES (38, 'Can change execute_the_program', 10, 'change_execute_the_program');
INSERT INTO `auth_permission` VALUES (39, 'Can delete execute_the_program', 10, 'delete_execute_the_program');
INSERT INTO `auth_permission` VALUES (40, 'Can view execute_the_program', 10, 'view_execute_the_program');
INSERT INTO `auth_permission` VALUES (41, 'Can add experimental_result', 11, 'add_experimental_result');
INSERT INTO `auth_permission` VALUES (42, 'Can change experimental_result', 11, 'change_experimental_result');
INSERT INTO `auth_permission` VALUES (43, 'Can delete experimental_result', 11, 'delete_experimental_result');
INSERT INTO `auth_permission` VALUES (44, 'Can view experimental_result', 11, 'view_experimental_result');
INSERT INTO `auth_permission` VALUES (45, 'Can add eyi_result', 12, 'add_eyi_result');
INSERT INTO `auth_permission` VALUES (46, 'Can change eyi_result', 12, 'change_eyi_result');
INSERT INTO `auth_permission` VALUES (47, 'Can delete eyi_result', 12, 'delete_eyi_result');
INSERT INTO `auth_permission` VALUES (48, 'Can view eyi_result', 12, 'view_eyi_result');
INSERT INTO `auth_permission` VALUES (49, 'Can add malicious', 13, 'add_malicious');
INSERT INTO `auth_permission` VALUES (50, 'Can change malicious', 13, 'change_malicious');
INSERT INTO `auth_permission` VALUES (51, 'Can delete malicious', 13, 'delete_malicious');
INSERT INTO `auth_permission` VALUES (52, 'Can view malicious', 13, 'view_malicious');
INSERT INTO `auth_permission` VALUES (53, 'Can add malicious_models_manage', 14, 'add_malicious_models_manage');
INSERT INTO `auth_permission` VALUES (54, 'Can change malicious_models_manage', 14, 'change_malicious_models_manage');
INSERT INTO `auth_permission` VALUES (55, 'Can delete malicious_models_manage', 14, 'delete_malicious_models_manage');
INSERT INTO `auth_permission` VALUES (56, 'Can view malicious_models_manage', 14, 'view_malicious_models_manage');
INSERT INTO `auth_permission` VALUES (57, 'Can add malicious_traffic11', 15, 'add_malicious_traffic11');
INSERT INTO `auth_permission` VALUES (58, 'Can change malicious_traffic11', 15, 'change_malicious_traffic11');
INSERT INTO `auth_permission` VALUES (59, 'Can delete malicious_traffic11', 15, 'delete_malicious_traffic11');
INSERT INTO `auth_permission` VALUES (60, 'Can view malicious_traffic11', 15, 'view_malicious_traffic11');
INSERT INTO `auth_permission` VALUES (61, 'Can add model_info', 16, 'add_model_info');
INSERT INTO `auth_permission` VALUES (62, 'Can change model_info', 16, 'change_model_info');
INSERT INTO `auth_permission` VALUES (63, 'Can delete model_info', 16, 'delete_model_info');
INSERT INTO `auth_permission` VALUES (64, 'Can view model_info', 16, 'view_model_info');
INSERT INTO `auth_permission` VALUES (65, 'Can add 模型管理', 17, 'add_modelmanagement');
INSERT INTO `auth_permission` VALUES (66, 'Can change 模型管理', 17, 'change_modelmanagement');
INSERT INTO `auth_permission` VALUES (67, 'Can delete 模型管理', 17, 'delete_modelmanagement');
INSERT INTO `auth_permission` VALUES (68, 'Can view 模型管理', 17, 'view_modelmanagement');
INSERT INTO `auth_permission` VALUES (69, 'Can add models_manage', 18, 'add_models_manage');
INSERT INTO `auth_permission` VALUES (70, 'Can change models_manage', 18, 'change_models_manage');
INSERT INTO `auth_permission` VALUES (71, 'Can delete models_manage', 18, 'delete_models_manage');
INSERT INTO `auth_permission` VALUES (72, 'Can view models_manage', 18, 'view_models_manage');
INSERT INTO `auth_permission` VALUES (73, 'Can add test_dataset', 19, 'add_test_dataset');
INSERT INTO `auth_permission` VALUES (74, 'Can change test_dataset', 19, 'change_test_dataset');
INSERT INTO `auth_permission` VALUES (75, 'Can delete test_dataset', 19, 'delete_test_dataset');
INSERT INTO `auth_permission` VALUES (76, 'Can view test_dataset', 19, 'view_test_dataset');
INSERT INTO `auth_permission` VALUES (77, 'Can add user', 20, 'add_userinfo');
INSERT INTO `auth_permission` VALUES (78, 'Can change user', 20, 'change_userinfo');
INSERT INTO `auth_permission` VALUES (79, 'Can delete user', 20, 'delete_userinfo');
INSERT INTO `auth_permission` VALUES (80, 'Can view user', 20, 'view_userinfo');
INSERT INTO `auth_permission` VALUES (81, 'Can add 检测历史', 21, 'add_detectionhistory');
INSERT INTO `auth_permission` VALUES (82, 'Can change 检测历史', 21, 'change_detectionhistory');
INSERT INTO `auth_permission` VALUES (83, 'Can delete 检测历史', 21, 'delete_detectionhistory');
INSERT INTO `auth_permission` VALUES (84, 'Can view 检测历史', 21, 'view_detectionhistory');
INSERT INTO `auth_permission` VALUES (85, 'Can add data augmentation task', 22, 'add_dataaugmentationtask');
INSERT INTO `auth_permission` VALUES (86, 'Can change data augmentation task', 22, 'change_dataaugmentationtask');
INSERT INTO `auth_permission` VALUES (87, 'Can delete data augmentation task', 22, 'delete_dataaugmentationtask');
INSERT INTO `auth_permission` VALUES (88, 'Can view data augmentation task', 22, 'view_dataaugmentationtask');
INSERT INTO `auth_permission` VALUES (89, 'Access admin page', 23, 'view');

-- ----------------------------
-- Table structure for database_info
-- ----------------------------
DROP TABLE IF EXISTS `database_info`;
CREATE TABLE `database_info`  (
  `database_id` int NOT NULL AUTO_INCREMENT,
  `database_name` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `database_grouping` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `database_instances` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `database_features` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `create_time` datetime(6) NOT NULL,
  PRIMARY KEY (`database_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of database_info
-- ----------------------------

-- ----------------------------
-- Table structure for database_manage2
-- ----------------------------
DROP TABLE IF EXISTS `database_manage2`;
CREATE TABLE `database_manage2`  (
  `Database_id` int NOT NULL AUTO_INCREMENT,
  `Database_name` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Database_number` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Database_type` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `create_time` datetime(6) NOT NULL,
  PRIMARY KEY (`Database_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of database_manage2
-- ----------------------------

-- ----------------------------
-- Table structure for django_admin_log
-- ----------------------------
DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE `django_admin_log`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  `object_repr` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `action_flag` smallint UNSIGNED NOT NULL,
  `change_message` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `content_type_id` int NULL DEFAULT NULL,
  `user_id` bigint NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `django_admin_log_content_type_id_c4bce8eb_fk_django_co`(`content_type_id` ASC) USING BTREE,
  INDEX `django_admin_log_user_id_c564eba6_fk_MTD_userinfo_id`(`user_id` ASC) USING BTREE,
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_MTD_userinfo_id` FOREIGN KEY (`user_id`) REFERENCES `mtd_userinfo` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `django_admin_log_chk_1` CHECK (`action_flag` >= 0)
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_admin_log
-- ----------------------------

-- ----------------------------
-- Table structure for django_content_type
-- ----------------------------
DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE `django_content_type`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `model` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `django_content_type_app_label_model_76bd3d3b_uniq`(`app_label` ASC, `model` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 24 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_content_type
-- ----------------------------
INSERT INTO `django_content_type` VALUES (1, 'admin', 'logentry');
INSERT INTO `django_content_type` VALUES (3, 'auth', 'group');
INSERT INTO `django_content_type` VALUES (2, 'auth', 'permission');
INSERT INTO `django_content_type` VALUES (4, 'contenttypes', 'contenttype');
INSERT INTO `django_content_type` VALUES (23, 'django_rq', 'queue');
INSERT INTO `django_content_type` VALUES (22, 'MTD', 'dataaugmentationtask');
INSERT INTO `django_content_type` VALUES (6, 'MTD', 'database_manage');
INSERT INTO `django_content_type` VALUES (7, 'MTD', 'database_manage2');
INSERT INTO `django_content_type` VALUES (8, 'MTD', 'datasetmanagement');
INSERT INTO `django_content_type` VALUES (21, 'MTD', 'detectionhistory');
INSERT INTO `django_content_type` VALUES (9, 'MTD', 'early_warning_database');
INSERT INTO `django_content_type` VALUES (10, 'MTD', 'execute_the_program');
INSERT INTO `django_content_type` VALUES (11, 'MTD', 'experimental_result');
INSERT INTO `django_content_type` VALUES (12, 'MTD', 'eyi_result');
INSERT INTO `django_content_type` VALUES (13, 'MTD', 'malicious');
INSERT INTO `django_content_type` VALUES (14, 'MTD', 'malicious_models_manage');
INSERT INTO `django_content_type` VALUES (15, 'MTD', 'malicious_traffic11');
INSERT INTO `django_content_type` VALUES (16, 'MTD', 'model_info');
INSERT INTO `django_content_type` VALUES (17, 'MTD', 'modelmanagement');
INSERT INTO `django_content_type` VALUES (18, 'MTD', 'models_manage');
INSERT INTO `django_content_type` VALUES (19, 'MTD', 'test_dataset');
INSERT INTO `django_content_type` VALUES (20, 'MTD', 'userinfo');
INSERT INTO `django_content_type` VALUES (5, 'sessions', 'session');

-- ----------------------------
-- Table structure for django_migrations
-- ----------------------------
DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE `django_migrations`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `app` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 26 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_migrations
-- ----------------------------
INSERT INTO `django_migrations` VALUES (1, 'contenttypes', '0001_initial', '2025-04-07 14:21:35.046317');
INSERT INTO `django_migrations` VALUES (2, 'contenttypes', '0002_remove_content_type_name', '2025-04-07 14:21:35.071376');
INSERT INTO `django_migrations` VALUES (3, 'auth', '0001_initial', '2025-04-07 14:21:35.159232');
INSERT INTO `django_migrations` VALUES (4, 'auth', '0002_alter_permission_name_max_length', '2025-04-07 14:21:35.188243');
INSERT INTO `django_migrations` VALUES (5, 'auth', '0003_alter_user_email_max_length', '2025-04-07 14:21:35.193232');
INSERT INTO `django_migrations` VALUES (6, 'auth', '0004_alter_user_username_opts', '2025-04-07 14:21:35.197232');
INSERT INTO `django_migrations` VALUES (7, 'auth', '0005_alter_user_last_login_null', '2025-04-07 14:21:35.201241');
INSERT INTO `django_migrations` VALUES (8, 'auth', '0006_require_contenttypes_0002', '2025-04-07 14:21:35.203231');
INSERT INTO `django_migrations` VALUES (9, 'auth', '0007_alter_validators_add_error_messages', '2025-04-07 14:21:35.208232');
INSERT INTO `django_migrations` VALUES (10, 'auth', '0008_alter_user_username_max_length', '2025-04-07 14:21:35.211239');
INSERT INTO `django_migrations` VALUES (11, 'auth', '0009_alter_user_last_name_max_length', '2025-04-07 14:21:35.215240');
INSERT INTO `django_migrations` VALUES (12, 'auth', '0010_alter_group_name_max_length', '2025-04-07 14:21:35.223480');
INSERT INTO `django_migrations` VALUES (13, 'auth', '0011_update_proxy_permissions', '2025-04-07 14:21:35.227479');
INSERT INTO `django_migrations` VALUES (14, 'auth', '0012_alter_user_first_name_max_length', '2025-04-07 14:21:35.231989');
INSERT INTO `django_migrations` VALUES (15, 'MTD', '0001_initial', '2025-04-07 14:21:35.482580');
INSERT INTO `django_migrations` VALUES (16, 'admin', '0001_initial', '2025-04-07 14:21:35.534696');
INSERT INTO `django_migrations` VALUES (17, 'admin', '0002_logentry_remove_auto_add', '2025-04-07 14:21:35.539694');
INSERT INTO `django_migrations` VALUES (18, 'admin', '0003_logentry_add_action_flag_choices', '2025-04-07 14:21:35.544694');
INSERT INTO `django_migrations` VALUES (19, 'sessions', '0001_initial', '2025-04-07 14:21:35.560686');
INSERT INTO `django_migrations` VALUES (20, 'MTD', '0002_alter_datasetmanagement_category_and_more', '2025-04-09 14:56:04.695252');
INSERT INTO `django_migrations` VALUES (21, 'MTD', '0003_dataaugmentationtask', '2025-04-10 09:10:06.254920');
INSERT INTO `django_migrations` VALUES (22, 'MTD', '0004_alter_dataaugmentationtask_log_file_path', '2025-04-10 09:16:43.309554');
INSERT INTO `django_migrations` VALUES (23, 'django_rq', '0001_initial', '2025-04-10 14:20:36.345777');
INSERT INTO `django_migrations` VALUES (24, 'MTD', '0005_auto_20250411_0925', '2025-04-11 01:25:52.037811');
INSERT INTO `django_migrations` VALUES (25, 'MTD', '0006_auto_20250412_2022', '2025-04-12 12:22:10.633648');

-- ----------------------------
-- Table structure for django_session
-- ----------------------------
DROP TABLE IF EXISTS `django_session`;
CREATE TABLE `django_session`  (
  `session_key` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `session_data` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`) USING BTREE,
  INDEX `django_session_expire_date_a5c62663`(`expire_date` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_session
-- ----------------------------
INSERT INTO `django_session` VALUES ('6m2guq4p90w4m6zwk2nq4qa0s9n1372j', '.eJxVjMsOwiAQRf-FtSHAMDxcuvcbyPCoVA0kpV0Z_12bdKHbe865LxZoW2vYRlnCnNmZWXb63SKlR2k7yHdqt85Tb-syR74r_KCDX3suz8vh_h1UGvVbIwgURMkgKY9STc5S1CZKaXMGNXlnELwuPpFAhcYpJAAVwUupo0ns_QHAJjar:1uQ1op:10OWPUHgjgzDy4Zn33LVI4YhXHKPRrIgd5XwUkZrA-Y', '2025-06-27 10:39:31.764170');
INSERT INTO `django_session` VALUES ('83ghz2kirzquntc1ai8li46ggyap1znq', '.eJxVjEEOwiAQAP_C2RB22QJ69O4bCCxUqgaS0p6MfzckPeh1ZjJv4cO-Fb_3vPoliYsAcfplMfAz1yHSI9R7k9zqti5RjkQetstbS_l1Pdq_QQm9jC1qUmdkl7LWju3MFNFGjZNNrGciYycgUM6gBjDIymRnLeQE0SAp8fkCspc2Qw:1u3Bz3:3a35cvHdZUZRemxpbIau0IeOU6gW6npPtEEni_oE9Sc', '2025-04-25 10:51:41.705463');
INSERT INTO `django_session` VALUES ('cc5pxzqeye155dgf8865j2pn2wmc7y8t', '.eJyNlE1v2zAMhv-Lz2UgUdRXb2t7XIFhbc8GRUmNt9Yp7ORU9L9PwbItwxzMN0Hiw_elSOm96_mw3_aHuUz9kLvrTndX53uJ5XsZjwf5G4_Pu43sxv00pM0xZHM6nTf3u1xebk6xfyXY8rw9pkVDKqKEXIwJ4qtQQp8MWp_FVCLnrSatgkOjtUNRrgTvdck6OSTVksphmsq47zPveS77fuTX0jLfPj7B7W4YX4exTGdhb9PueSrz3F1rpa66_cTDOIzP_RtP_Np237v_JHqZjr4LUFuPu2Eu_Szbkg8vx2DZzS2s-7jqfun0xK0eLwGccQ6IjIfEWAGNNyFYJAzxZOY3wylpwWigltqYTAxJ2abJURdC5arHE3PudiXXPT083sLDdqiH7ky0Brbso4OqjkZVbEZdyKAVSaYchQwtia7jLogqUaSKJ0giGchXBiZnIceoNJHPmdOS6Drun-b96YpoVckR6OI9kG2mA1IEa21CSVbaMC7pruMuFEsVdZtugsylmQ41QMKqQKNgm3tN1i22dR13udjKjJnFA-efw9BWjkurQWHxUSpms9jZVdyFYlPWtWabwAcrQKUWYEMekK0SJ4VFFsdpHXdBNLCjyNLG3mvVHBcEZouga_tjMKGLui6JruMu37COlEJNBQxq295AksYbgaIpp5B95GSXdNdx3c3w-eHx_lwQvUs5mwQmt-dOPllIOlTwnJTGhhW9KLiO6-7uvsDdp6_dxw_JB82n:1u3C1Y:BnUQIs3-nZ_VOpPG90x8XumCcK_omdsYfgOfGULSXj8', '2025-04-25 10:54:16.826573');
INSERT INTO `django_session` VALUES ('ywt4us5c3rxqaf35okkr5yr99ynlhg5d', '.eJyNjDuSgzAQBe-iGKgZBBqNw914fQXVII2M9wNbCEcu391Q5cDOnL7uflcT5LKO4VJ0CedkDqY31fM2SPzRaQfpW6bT3MR5Wpfz0OxK86Cl-ZqT_n483JeDUcq41Ug4kHpsCURti-Ahq09MLmbyqNQ7zhzBtop98hbZRbt5NqMFaXE7_V_m06KlBOg8RexcTYJUd4Jc8wB9LcycO6uKzpoDAlQmySpF1zDJn77Zmc_j0dzuuARUJQ:1u7tSb:Ak3HJhILXHg8Z_OghgTBE1nB2UC3Xugixb-osMvZzis', '2025-05-08 10:05:37.442700');

-- ----------------------------
-- Table structure for early_warning_database
-- ----------------------------
DROP TABLE IF EXISTS `early_warning_database`;
CREATE TABLE `early_warning_database`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `tool_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `testcase_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_a` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_p` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_r` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_f` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `statu` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of early_warning_database
-- ----------------------------

-- ----------------------------
-- Table structure for execute_the_program
-- ----------------------------
DROP TABLE IF EXISTS `execute_the_program`;
CREATE TABLE `execute_the_program`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `path` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of execute_the_program
-- ----------------------------

-- ----------------------------
-- Table structure for experimental_result
-- ----------------------------
DROP TABLE IF EXISTS `experimental_result`;
CREATE TABLE `experimental_result`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `tool_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `testcase_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_a` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_p` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_r` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `indicator_f` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of experimental_result
-- ----------------------------

-- ----------------------------
-- Table structure for eyi_result
-- ----------------------------
DROP TABLE IF EXISTS `eyi_result`;
CREATE TABLE `eyi_result`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `models_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `database_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `average_a` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `average_p` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `average_r` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `average_f` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of eyi_result
-- ----------------------------

-- ----------------------------
-- Table structure for malicious
-- ----------------------------
DROP TABLE IF EXISTS `malicious`;
CREATE TABLE `malicious`  (
  `Mid` int NOT NULL AUTO_INCREMENT,
  `Mname` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Mpath` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`Mid`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of malicious
-- ----------------------------
INSERT INTO `malicious` VALUES (1, 'CWE', 'TEST');

-- ----------------------------
-- Table structure for malicious_models_manage
-- ----------------------------
DROP TABLE IF EXISTS `malicious_models_manage`;
CREATE TABLE `malicious_models_manage`  (
  `Malicious_model_id` int NOT NULL AUTO_INCREMENT,
  `Malicious_model_name` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `Malicious_model_grouping` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `create_time` datetime(6) NULL DEFAULT NULL,
  `is_Bidirectional` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `is_feature` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`Malicious_model_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of malicious_models_manage
-- ----------------------------

-- ----------------------------
-- Table structure for model_info
-- ----------------------------
DROP TABLE IF EXISTS `model_info`;
CREATE TABLE `model_info`  (
  `model_id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `model_info_url` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `upload_date` datetime(6) NOT NULL,
  PRIMARY KEY (`model_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of model_info
-- ----------------------------

-- ----------------------------
-- Table structure for models_info
-- ----------------------------
DROP TABLE IF EXISTS `models_info`;
CREATE TABLE `models_info`  (
  `model_id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `model_grouping` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `create_time` datetime(6) NULL DEFAULT NULL,
  `is_incremental_learning` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `is_multiple` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`model_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of models_info
-- ----------------------------

-- ----------------------------
-- Table structure for mtd_datasetmanagement
-- ----------------------------
DROP TABLE IF EXISTS `mtd_datasetmanagement`;
CREATE TABLE `mtd_datasetmanagement`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `category` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `upload_time` datetime(6) NOT NULL,
  `data_file` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `size` int UNSIGNED NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name` ASC) USING BTREE,
  CONSTRAINT `mtd_datasetmanagement_chk_1` CHECK (`size` >= 0)
) ENGINE = InnoDB AUTO_INCREMENT = 17 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_datasetmanagement
-- ----------------------------
INSERT INTO `mtd_datasetmanagement` VALUES (2, 'USTC-TFC2016', 'RGB', '2025-04-09 12:14:37.892462', 'datasets/USTC-TFC2016-master.zip', 390500);
INSERT INTO `mtd_datasetmanagement` VALUES (3, 'CTU', 'RGB', '2025-04-09 12:15:18.367627', 'datasets/CTU.zip', 143798);
INSERT INTO `mtd_datasetmanagement` VALUES (4, 'ISAC', 'RGB', '2025-04-09 12:15:45.015408', 'datasets/ISAC.zip', 229357);
INSERT INTO `mtd_datasetmanagement` VALUES (5, 'DDoSdata', 'CSV', '2025-04-09 12:20:16.738297', 'datasets/DDoSdata.csv', 211520);
INSERT INTO `mtd_datasetmanagement` VALUES (9, 'KDD-CUP', 'CSV', '2025-04-11 01:55:31.139401', 'datasets/KDDTest.csv', 561468);
INSERT INTO `mtd_datasetmanagement` VALUES (10, 'ISAC-PCAP', 'PCAP', '2025-04-11 03:03:09.655600', 'datasets/ISAC_WrJz68h.zip', 163432);
INSERT INTO `mtd_datasetmanagement` VALUES (11, 'CTU-PACP', 'PCAP', '2025-04-11 05:23:47.226775', 'datasets/CTU_3R2vNnn.zip', 293841);
INSERT INTO `mtd_datasetmanagement` VALUES (14, 'USTC-Enhanced ', 'RGB', '2025-04-15 13:54:47.072663', 'datasets/USTC.zip', 352814);

-- ----------------------------
-- Table structure for mtd_detectionhistory
-- ----------------------------
DROP TABLE IF EXISTS `mtd_detectionhistory`;
CREATE TABLE `mtd_detectionhistory`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `detection_time` datetime(6) NOT NULL,
  `is_malicious` tinyint(1) NOT NULL,
  `accuracy` decimal(5, 2) NULL DEFAULT NULL,
  `report` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `dataset_id` bigint NOT NULL,
  `model_id` bigint NOT NULL,
  `F1_score` decimal(5, 2) NULL DEFAULT NULL,
  `FPR` decimal(5, 2) NULL DEFAULT NULL,
  `TPR` decimal(5, 2) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `MTD_detectionhistory_dataset_id_18c9707c_fk_MTD_datas`(`dataset_id` ASC) USING BTREE,
  INDEX `MTD_detectionhistory_model_id_fe299141_fk_MTD_modelmanagement_id`(`model_id` ASC) USING BTREE,
  CONSTRAINT `MTD_detectionhistory_dataset_id_18c9707c_fk_MTD_datas` FOREIGN KEY (`dataset_id`) REFERENCES `mtd_datasetmanagement` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `MTD_detectionhistory_model_id_fe299141_fk_MTD_modelmanagement_id` FOREIGN KEY (`model_id`) REFERENCES `mtd_modelmanagement` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 71 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_detectionhistory
-- ----------------------------
INSERT INTO `mtd_detectionhistory` VALUES (23, '2025-04-11 04:59:37.950570', 0, 94.44, '', 2, 1, 93.96, 0.69, 93.49);
INSERT INTO `mtd_detectionhistory` VALUES (24, '2025-04-11 04:59:48.881139', 0, 93.91, '', 10, 1, 93.25, 0.53, 92.60);
INSERT INTO `mtd_detectionhistory` VALUES (25, '2025-04-11 05:00:52.594604', 0, 86.27, '', 2, 10, 86.15, 3.62, 86.04);
INSERT INTO `mtd_detectionhistory` VALUES (26, '2025-04-11 05:01:01.743740', 0, 92.10, '', 2, 9, 91.42, 0.87, 90.74);
INSERT INTO `mtd_detectionhistory` VALUES (27, '2025-04-11 05:01:13.857275', 1, 99.21, '', 2, 8, 99.20, 0.13, 98.35);
INSERT INTO `mtd_detectionhistory` VALUES (28, '2025-04-11 05:02:29.303714', 1, 96.82, '', 2, 6, 96.78, 0.41, 96.74);
INSERT INTO `mtd_detectionhistory` VALUES (29, '2025-04-11 05:02:37.964584', 1, 96.01, '', 2, 5, 95.88, 0.46, 95.74);
INSERT INTO `mtd_detectionhistory` VALUES (30, '2025-04-11 05:02:51.135589', 0, 86.53, '', 10, 2, 86.15, 3.18, 85.77);
INSERT INTO `mtd_detectionhistory` VALUES (31, '2025-04-11 05:02:59.002816', 0, 88.12, '', 2, 2, 87.70, 2.66, 87.29);
INSERT INTO `mtd_detectionhistory` VALUES (32, '2025-04-11 05:06:13.812521', 1, 98.17, '', 2, 11, 97.77, 0.31, 97.37);
INSERT INTO `mtd_detectionhistory` VALUES (33, '2025-04-11 05:10:41.589261', 1, 90.96, '', 9, 9, 90.11, 0.76, 89.28);
INSERT INTO `mtd_detectionhistory` VALUES (35, '2025-04-11 07:25:43.436147', 0, 95.16, '', 3, 19, 94.79, 0.99, 94.42);
INSERT INTO `mtd_detectionhistory` VALUES (36, '2025-04-11 07:26:00.521486', 0, 99.45, '', 3, 8, 98.64, 0.12, 97.85);
INSERT INTO `mtd_detectionhistory` VALUES (37, '2025-04-11 07:28:59.854612', 1, 97.11, '', 3, 16, 96.95, 0.16, 96.79);
INSERT INTO `mtd_detectionhistory` VALUES (38, '2025-04-11 07:29:12.872724', 0, 94.06, '', 3, 15, 94.04, 0.95, 94.03);
INSERT INTO `mtd_detectionhistory` VALUES (39, '2025-04-11 07:29:25.564217', 0, 95.82, '', 3, 14, 94.88, 0.41, 93.95);
INSERT INTO `mtd_detectionhistory` VALUES (40, '2025-04-11 07:29:43.189350', 0, 96.81, '', 3, 13, 96.26, 0.28, 95.72);
INSERT INTO `mtd_detectionhistory` VALUES (41, '2025-04-11 07:30:34.311106', 0, 95.76, '', 3, 19, 95.36, 1.14, 94.96);
INSERT INTO `mtd_detectionhistory` VALUES (42, '2025-04-11 07:31:03.022106', 1, 96.75, '', 3, 12, 96.22, 0.46, 95.69);
INSERT INTO `mtd_detectionhistory` VALUES (43, '2025-04-11 10:43:29.923333', 1, 96.63, '', 11, 16, 96.00, 0.14, 95.38);
INSERT INTO `mtd_detectionhistory` VALUES (44, '2025-04-11 10:44:34.267694', 1, 93.32, '', 11, 19, 92.61, 0.96, 91.91);
INSERT INTO `mtd_detectionhistory` VALUES (45, '2025-04-11 10:45:41.084756', 0, 92.56, '', 11, 19, 92.43, 1.26, 92.30);
INSERT INTO `mtd_detectionhistory` VALUES (46, '2025-04-11 10:46:17.049744', 1, 93.58, '', 11, 19, 93.24, 1.33, 92.90);
INSERT INTO `mtd_detectionhistory` VALUES (47, '2025-04-11 10:52:30.485279', 0, 95.46, '', 4, 19, 94.72, 1.31, 93.99);
INSERT INTO `mtd_detectionhistory` VALUES (49, '2025-04-12 02:53:32.675788', 1, 91.12, '', 10, 15, 90.86, 1.02, 90.60);
INSERT INTO `mtd_detectionhistory` VALUES (50, '2025-04-12 02:54:08.471460', 0, 91.44, '', 10, 15, 91.11, 0.94, 90.78);
INSERT INTO `mtd_detectionhistory` VALUES (51, '2025-04-12 11:14:43.139422', 0, 95.58, '', 4, 19, 94.58, 1.30, 93.61);
INSERT INTO `mtd_detectionhistory` VALUES (52, '2025-04-12 11:14:51.240811', 0, 96.36, '', 4, 18, 95.60, 0.59, 94.86);
INSERT INTO `mtd_detectionhistory` VALUES (53, '2025-04-12 11:15:00.788253', 0, 98.23, '', 4, 16, 97.71, 0.17, 97.20);
INSERT INTO `mtd_detectionhistory` VALUES (54, '2025-04-12 12:54:40.341136', 0, 92.06, '', 11, 15, 91.20, 1.23, 90.35);
INSERT INTO `mtd_detectionhistory` VALUES (55, '2025-04-12 14:01:57.504589', 0, 99.40, '', 5, 8, 98.45, 0.14, 97.52);
INSERT INTO `mtd_detectionhistory` VALUES (56, '2025-04-13 11:21:07.551615', 0, 97.39, '', 9, 13, 97.21, 0.39, 97.04);
INSERT INTO `mtd_detectionhistory` VALUES (57, '2025-04-13 13:22:19.177148', 1, 92.33, '', 2, 9, 92.26, 0.73, 92.20);
INSERT INTO `mtd_detectionhistory` VALUES (58, '2025-04-13 13:40:58.198799', 0, 99.45, '', 4, 8, 99.12, 0.12, 98.98);
INSERT INTO `mtd_detectionhistory` VALUES (59, '2025-04-13 13:41:10.550041', 0, 98.19, '', 4, 11, 97.82, 0.28, 97.44);
INSERT INTO `mtd_detectionhistory` VALUES (60, '2025-04-15 13:55:25.107400', 1, 99.99, '', 14, 8, 99.99, 0.09, 99.38);
INSERT INTO `mtd_detectionhistory` VALUES (61, '2025-04-15 13:55:36.401695', 0, 91.37, '', 14, 9, 90.36, 0.95, 89.38);
INSERT INTO `mtd_detectionhistory` VALUES (62, '2025-04-15 13:55:49.915959', 1, 94.31, '', 14, 4, 93.83, 0.70, 93.36);
INSERT INTO `mtd_detectionhistory` VALUES (63, '2025-04-15 13:55:59.366262', 1, 96.00, '', 14, 5, 95.14, 0.44, 94.30);
INSERT INTO `mtd_detectionhistory` VALUES (64, '2025-04-15 13:56:24.081134', 1, 93.42, '', 5, 1, 93.02, 0.57, 92.62);
INSERT INTO `mtd_detectionhistory` VALUES (65, '2025-04-15 13:56:34.907356', 0, 97.91, '', 5, 11, 97.19, 0.26, 96.48);
INSERT INTO `mtd_detectionhistory` VALUES (66, '2025-04-15 13:57:57.288754', 1, 97.57, '', 11, 8, 97.37, 0.14, 97.17);
INSERT INTO `mtd_detectionhistory` VALUES (67, '2025-04-15 13:58:06.448245', 1, 96.91, '', 10, 8, 96.73, 0.12, 96.56);
INSERT INTO `mtd_detectionhistory` VALUES (68, '2025-04-15 13:58:24.031444', 1, 99.32, '', 5, 8, 98.74, 0.12, 98.17);
INSERT INTO `mtd_detectionhistory` VALUES (69, '2025-04-15 13:58:32.532552', 1, 99.65, '', 9, 8, 98.77, 0.14, 97.91);

-- ----------------------------
-- Table structure for mtd_malicious_traffic11
-- ----------------------------
DROP TABLE IF EXISTS `mtd_malicious_traffic11`;
CREATE TABLE `mtd_malicious_traffic11`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `path` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_malicious_traffic11
-- ----------------------------

-- ----------------------------
-- Table structure for mtd_modelmanagement
-- ----------------------------
DROP TABLE IF EXISTS `mtd_modelmanagement`;
CREATE TABLE `mtd_modelmanagement`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `category` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `upload_time` datetime(6) NOT NULL,
  `model_file` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `description` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 24 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_modelmanagement
-- ----------------------------
INSERT INTO `mtd_modelmanagement` VALUES (1, 'CNN', 'CNN', '2025-04-09 21:54:48.000000', 'models/CNN.pt', 'CNN主要用于处理具有网格结构的数据,如图像；使用卷积层提取特征,池化层降维，擅长捕捉局部特征和空间关系，在图像识别、目标检测等任务中表现出色，但容易过拟合');
INSERT INTO `mtd_modelmanagement` VALUES (2, 'RNN', 'RNN', '2025-04-09 14:00:58.148767', 'models/RNN__final_0.pth', 'RNN循环神经网络设计用于处理序列数据\r\n有内部状态(记忆),可以记住之前的信息，适合处理时间序列数据,如自然语言处理，存在长期依赖问题');
INSERT INTO `mtd_modelmanagement` VALUES (4, 'LSTM', 'LSTM', '2025-04-09 14:17:51.030556', 'models/LSTM_model_best_CTU_0.pt', 'LSTM是RNN的一种变体,解决了长期依赖问题，有门控机制:输入门、遗忘门、输出门，能够长期保存重要信息,忘记不相关信息，在语音识别、机器翻译等任务中广泛应用');
INSERT INTO `mtd_modelmanagement` VALUES (5, 'TCN', 'TCN', '2025-04-09 14:19:02.919772', 'models/meta_TCN_model.pt', 'TCN模型结合了CNN和RNN的优点，使用膨胀卷积,可以捕捉长期依赖，并行计算效率高,训练速度较快，在时间序列预测、音频生成等任务中表现良好');
INSERT INTO `mtd_modelmanagement` VALUES (6, 'BiLSTM', 'BiLSTM', '2025-04-09 14:23:33.053559', 'models/BiLSTM_best_USTC.pt', 'LSTM的扩展,包含两个方向的LSTM，可以同时考虑过去和未来的上下文信息，在许多序列标注任务中表现优秀');
INSERT INTO `mtd_modelmanagement` VALUES (8, 'DMSE (Deep-Multi-Stacking-Ensemble)', 'DMSE', '2025-04-09 14:49:48.822606', 'models/Meta_BiTCN_best_USTC_1.pth', '使用CNN、LSTM、TCN、BiLSTM、BiTCN作为基学习器，从时序和空间卷积两方面充分学习网络流量的特征，使用多堆叠集成作为集成学习的集成策略，极大的增强检测模型的泛化能力和稳定性，使用BiTCN作为二层堆叠集成的元学习器，可以学习到流量的高维时序和空间特征，增强对流量的识别效果和精确度。');
INSERT INTO `mtd_modelmanagement` VALUES (9, 'ResNet', 'Deep_Learning', '2025-04-09 14:56:14.012121', 'models/Resnet34_CTU_0_X2FV4ru.pth', 'ResNet（Residual Network）是一种深度卷积神经网络架构，旨在解决深度学习中的梯度消失和模型训练困难等问题。ResNet 的核心创新是在网络中引入了“残差连接”（skip connections），使得信息能够在网络中直接流动。');
INSERT INTO `mtd_modelmanagement` VALUES (10, 'EfficientNet', 'Deep_Learning', '2025-04-09 14:57:34.473932', 'models/Efficientnet_model_final_USTC_0.pth', 'EfficientNet 是一种高效的卷积神经网络架构，首次由 Google 研究团队于 2019 年提出。其主要目标是通过优化网络的深度（Depth）、宽度（Width）和分辨率（Resolution）来实现更高的效率与准确率，从而取得最佳的模型性能。');
INSERT INTO `mtd_modelmanagement` VALUES (11, 'BiTCN', 'BiTCN', '2025-04-11 05:04:54.761522', 'models/BiTCN_best_USTC_8.pt', '双向时序卷积网络');
INSERT INTO `mtd_modelmanagement` VALUES (12, 'Meta-CNN', 'CNN', '2025-04-11 05:32:33.285917', 'models/meta_CNN_best_CTU_4.pth', 'CNN作为stacking堆叠集成的第二层元学习器的模型');
INSERT INTO `mtd_modelmanagement` VALUES (13, 'Meta-BiLSTM', 'BiLSTM', '2025-04-11 05:33:29.699797', 'models/meta_BiLSTM_best_6.pth', 'BiLSTM模型作为stacking堆叠集成的第二层元学习器的模型');
INSERT INTO `mtd_modelmanagement` VALUES (14, 'Meta-LSTM', 'LSTM', '2025-04-11 05:34:01.901087', 'models/meta_LSTM_best_5.pt', 'LSTM模型作为stacking堆叠集成的第二层元学习器的模型');
INSERT INTO `mtd_modelmanagement` VALUES (15, 'Meta-EfficientNet', 'Machine_Learning', '2025-04-11 05:35:32.238677', 'models/meta_Efficientnet_model_best_CTU_5.pth', 'EfficientNet模型作为stacking堆叠集成的第二层元学习器的模型');
INSERT INTO `mtd_modelmanagement` VALUES (16, 'Meta-TCN', 'TCN', '2025-04-11 05:36:14.935361', 'models/Meta_TCN_best_CTU_4.pth', 'TCN模型作为stacking堆叠集成的第二层元学习器的模型');
INSERT INTO `mtd_modelmanagement` VALUES (18, 'Meta-ResNet', 'Deep_Learning', '2025-04-11 06:24:30.786689', 'models/meta_ResNet_model_best_9.pth', 'ResNet模型作为stacking堆叠集成的第二层元学习器的模型');
INSERT INTO `mtd_modelmanagement` VALUES (19, 'Meta-RNN', 'RNN', '2025-04-11 06:24:51.638662', 'models/meta_RNN_CTU_best_9.pth', 'RNN模型作为stacking堆叠集成的第二层元学习器的模型');

-- ----------------------------
-- Table structure for mtd_userinfo
-- ----------------------------
DROP TABLE IF EXISTS `mtd_userinfo`;
CREATE TABLE `mtd_userinfo`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `password` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `last_login` datetime(6) NULL DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `first_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `last_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `email` varchar(254) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  `phone` varchar(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `sex` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `birth` date NULL DEFAULT NULL,
  `avatar` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 8 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_userinfo
-- ----------------------------
INSERT INTO `mtd_userinfo` VALUES (1, 'pbkdf2_sha256$260000$sbspRhBjjw7Z0n2CDq8Jfo$bXlL4tKb8xoE+DNcnlpSbsH0NGAMDxfXkpjOQReufQA=', '2025-04-17 14:01:55.662101', 0, 'Solitude_zy', 'Yang', 'z', 'q2205773452@163.com', 0, 1, '2025-04-07 14:22:48.726813', '19599965629', 'M', '2003-01-19', 'avatars/猫猫1.jpeg');
INSERT INTO `mtd_userinfo` VALUES (2, 'pbkdf2_sha256$260000$1NBCe7vNro8m4wqjXHQHcl$qmG/C8W9TM9jWc6lnwCJf3qqHsMTD5KNjHoHAQ1IuD8=', '2025-04-12 13:12:22.716247', 0, 'zy', '', '', 'q2205773452@163.com', 0, 1, '2025-04-12 13:12:22.598386', '', '', NULL, '');
INSERT INTO `mtd_userinfo` VALUES (3, 'pbkdf2_sha256$260000$dGXRsyQ5QER37Qb99DYuKv$thGpHoSwLTuihFTWZpFKCNZrMClC8N1Da4zcHaOC4O0=', '2025-04-12 13:13:01.607169', 0, 'WYY', '', '', 'xztszy@gmail.com', 0, 1, '2025-04-12 13:13:01.495981', '', '', NULL, '');
INSERT INTO `mtd_userinfo` VALUES (4, 'pbkdf2_sha256$260000$uQXa6PYCpLBjt5EemO3X0g$ewt0RCtNc29trI28A7ns08pF+UyXzFhMPt5xM7UbvdU=', '2025-04-13 12:54:34.124722', 0, 'wy', '', '', 'xztszy@gmail.com', 0, 1, '2025-04-12 13:17:37.504705', '', '', NULL, 'avatars/猫猫2.jpeg');
INSERT INTO `mtd_userinfo` VALUES (5, 'pbkdf2_sha256$260000$dDCZ9FYmP2bcuqT9DQw3hS$2S8YVsAU+tzUuvPSmMnPvwdYMv5wvAGO9UTz4uNUuW4=', '2025-05-15 07:16:33.807023', 0, 'mr', '', '', 'q2205773452@163.com', 0, 1, '2025-04-24 10:03:39.048567', '', '', NULL, '');
INSERT INTO `mtd_userinfo` VALUES (6, 'pbkdf2_sha256$260000$UnBwl0fVWr2sI5OXELBWqx$lzpKMiG6XINmZ7AhaAs+8mgnA61TYZB8VwHj5N/jWv4=', '2025-05-15 07:35:29.268790', 0, 'zyy', '', '', 'xztszy@gmail.com', 0, 1, '2025-05-15 07:35:29.118099', '', '', NULL, '');
INSERT INTO `mtd_userinfo` VALUES (7, 'pbkdf2_sha256$260000$KcTyMBaI8frGv7PnMuOziE$b4DdX+sqW83omQOFY1A8OpviWJDpudAZoO6hSi/SBgU=', '2025-06-13 10:39:31.760168', 0, 'zzyy', '', '', 'qq@163.com', 0, 1, '2025-05-15 07:53:42.711736', '', '', NULL, 'avatars/心晴简单.jpeg');

-- ----------------------------
-- Table structure for mtd_userinfo_groups
-- ----------------------------
DROP TABLE IF EXISTS `mtd_userinfo_groups`;
CREATE TABLE `mtd_userinfo_groups`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userinfo_id` bigint NOT NULL,
  `group_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `MTD_userinfo_groups_userinfo_id_group_id_849b9305_uniq`(`userinfo_id` ASC, `group_id` ASC) USING BTREE,
  INDEX `MTD_userinfo_groups_group_id_4b76e195_fk_auth_group_id`(`group_id` ASC) USING BTREE,
  CONSTRAINT `MTD_userinfo_groups_group_id_4b76e195_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `MTD_userinfo_groups_userinfo_id_b3d45e72_fk_MTD_userinfo_id` FOREIGN KEY (`userinfo_id`) REFERENCES `mtd_userinfo` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_userinfo_groups
-- ----------------------------

-- ----------------------------
-- Table structure for mtd_userinfo_user_permissions
-- ----------------------------
DROP TABLE IF EXISTS `mtd_userinfo_user_permissions`;
CREATE TABLE `mtd_userinfo_user_permissions`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userinfo_id` bigint NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `MTD_userinfo_user_permis_userinfo_id_permission_i_8fdc5d9d_uniq`(`userinfo_id` ASC, `permission_id` ASC) USING BTREE,
  INDEX `MTD_userinfo_user_pe_permission_id_9b8fedac_fk_auth_perm`(`permission_id` ASC) USING BTREE,
  CONSTRAINT `MTD_userinfo_user_pe_permission_id_9b8fedac_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `MTD_userinfo_user_pe_userinfo_id_d7fc6a7c_fk_MTD_useri` FOREIGN KEY (`userinfo_id`) REFERENCES `mtd_userinfo` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of mtd_userinfo_user_permissions
-- ----------------------------

-- ----------------------------
-- Table structure for test_dataset
-- ----------------------------
DROP TABLE IF EXISTS `test_dataset`;
CREATE TABLE `test_dataset`  (
  `test_id` int NOT NULL AUTO_INCREMENT,
  `test_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `test_path` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `upload_date` datetime(6) NOT NULL,
  PRIMARY KEY (`test_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of test_dataset
-- ----------------------------

SET FOREIGN_KEY_CHECKS = 1;
