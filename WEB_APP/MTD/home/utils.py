import os
import shutil
from tokenize import group
from django.forms import model_to_dict
import json
import ast
import re
from django.http import HttpResponse
# from apps import student
from .. import models
from django.utils import timezone
from django.db.models import F, Avg
from APP_core import settings


def check_safeinput(str):
    safe_chars = ['&', '/', '\\', '?', '*', '#', '@', '|', ' ']
    for c in safe_chars:
        if c in str:
            return True
    return False


# 学生端-仅获取学生报告信息
def get_report_status(mchapter_id, mstudent_id):
    flag = models.Report.objects.filter(student=mstudent_id, chapter=mchapter_id)
    if flag.count() > 0:
        return flag[0].score
    else:
        return -2


# 学生端-course_demo获取所有章节的状态
def get_chapter_all_status(mchapter_id, mstudent_id):
    res = {"tag1": 0, "tag2": 0, "tag3": 0, "tag4": 0, "tag5": 0, "tag6": 0, "tag7": 0, "tag8": 0}
    flag = models.Learning_record.objects.filter(student=mstudent_id, chapter=mchapter_id)
    if flag.count() > 0:
        res["tag1"] = flag[0].tag1
        res["tag2"] = flag[0].tag2
        res["tag3"] = flag[0].tag3
        res["tag4"] = flag[0].tag4
        res["tag5"] = flag[0].tag5
        res["tag6"] = flag[0].tag6
        res["tag7"] = flag[0].tag7
        res["tag8"] = flag[0].tag8
    else:
        models.Learning_record.objects.create(tag1=0, tag2=0, tag3=0, tag4=0, tag5=0,
                                              tag6=0, tag7=0, tag8=0, chapter_id=mchapter_id, student_id=mstudent_id)
    return res


# 获取注册审核信息
def get_registration_audit():
    audit_list = []
    audit_data = models.UserInfo.objects.filter(is_active=0)
    for i in audit_data:
        index = model_to_dict(i)
        index['date_joined'] = index['date_joined'].strftime('%Y-%m-%d %H:%M:%S')
        audit_list.append(index)
        del index['password'], index['last_login'], index['is_superuser'], index['is_active'], index[
            'is_staff']
    return audit_list


# 获取教师与学生的注册审核信息
def get_registration_all():
    audit_list = []
    audit_data = models.UserInfo.objects.filter(is_superuser=0).filter(is_active=0)
    for i in audit_data:
        class_name = i.classroom.class_name
        user = i.to_dict()
        user["class_name"] = class_name
        user["birth"] = user["birth"].strftime('%Y-%m-%d')
        if user["is_staff"]:
            user["is_staff"] = "教师"
        else:
            user["is_staff"] = "学生"
        # index = model_to_dict(i)
        # index['date_joined'] = index['date_joined'].strftime('%Y-%m-%d %H:%M:%S')
        del user['password'], user['last_login'], user['is_superuser'], user['is_active']
        audit_list.append(user)
    return audit_list


# 审核用户 permission -1拒绝(删除用户) 1学生 2教师 3管理员
def audit_user(muser_id, permission):
    try:
        if permission == -1:
            models.UserInfo.objects.filter(id=muser_id).delete()
            return muser_id
        if permission == -11:
            models.UserInfo.objects.filter(is_active=0).delete()
            return 1
        if permission == 11:
            models.UserInfo.objects.filter(is_active=0).update(is_active=1)
            return 1
        muser = models.UserInfo.objects.get(id=muser_id)
        if permission == 1:
            # models.UserInfo.objects.filter(id=muser_id).update(is_active=1)
            muser.is_active = 1
        if permission == 2:
            muser.is_staff = 1
        if permission == 3:
            muser.is_superuser = 1
        muser.save()
    except Exception as e:
        print(e)
    return muser_id


# 修改用户信息
def update_userinfo(muser_id, info):
    try:
        muser = models.UserInfo.objects.get(id=muser_id)
        muser.username = info.username
        muser.first_name = info.first_name
        muser.email = info.email
        muser.phone = info.phone
        muser.sex = muser.sex
        muser.save()
        return 1
    except Exception as e:
        print(e)
    return 0


# 获取学生列表
def get_studentinfo():
    students = models.UserInfo.objects.filter(is_staff=1)
    return students


# 获取教师用户
def get_teacherinfo():
    teachers = models.UserInfo.objects.filter(is_staff=2)
    return teachers


# 获取班级名
def get_class_data(tag=0):
    res = []
    if tag == 1:
        data = models.Classinfo.objects.filter(grouping=0)
    else:
        data = models.Classinfo.objects.all()
    for i in data:
        res.append(i.class_name)
    return res


# ################################################################################################################################################################
# 获取模型名
def get_model_info_all():
    res = []
    data = models.Models_manage.objects.all()
    for i in data:
        temp = {}
        temp['model_id'] = i.model_id
        temp['model_name'] = i.model_name
        temp['model_grouping'] = i.model_grouping
        temp['is_incremental_learning'] = i.is_incremental_learning
        temp['is_multiple'] = i.is_multiple
        temp['create_time'] = i.create_time.strftime('%Y-%m-%d %H:%M:%S')
        res.append(temp)
    return res


def get_eyimodel_info_all():
    res = []
    data = models.eyimodels_manage.objects.all()
    for i in data:
        temp = {}
        temp['eyimodel_id'] = i.eyimodel_id
        temp['eyimodel_name'] = i.eyimodel_name
        temp['eyimodel_grouping'] = i.eyimodel_grouping
        temp['is_shangxiang'] = i.is_shangxiang
        temp['is_tezheng'] = i.is_tezheng
        temp['create_time'] = i.create_time.strftime('%Y-%m-%d %H:%M:%S')
        res.append(temp)
    return res


def get_record_info_all():
    res = []
    data = models.experimental_result.objects.all()
    for i in data:
        temp = {}
        temp['id'] = i.id
        temp['tool_name'] = i.tool_name
        temp['testcase_name'] = i.testcase_name
        temp['indicator_a'] = i.indicator_a
        temp['indicator_p'] = i.indicator_p
        temp['indicator_r'] = i.indicator_r
        temp['indicator_f'] = i.indicator_f
        res.append(temp)
    return res


def get_eyirecord_info_all():
    res = []
    data = models.eyi_result.objects.all()
    for i in data:
        temp = {}
        temp['id'] = i.id
        temp['models_name'] = i.models_name
        temp['database_name'] = i.database_name
        temp['average_a'] = i.average_a
        temp['average_p'] = i.average_p
        temp['average_r'] = i.average_r
        temp['average_f'] = i.average_f
        res.append(temp)
    return res


def get_shangchuan_info_all():
    res = []
    data = models.Execute_the_program.objects.all()
    for i in data:
        temp = {}
        temp['id'] = i.id
        temp['name'] = i.name
        temp['path'] = i.path
        res.append(temp)
    return res


def malicious_traffic_all():
    res = []
    data = models.malicious.objects.all()
    for i in data:
        temp = {}
        temp['Mid'] = i.Mid
        temp['Mname'] = i.Mname
        temp['Mpath'] = i.Mpath
        res.append(temp)
    return res


# 获得模型
def get_class_info_all():
    res = []
    data = models.Classinfo.objects.all()
    for i in data:
        temp = {}
        temp['class_id'] = i.class_id
        temp['class_name'] = i.class_name
        temp['student_numbers'] = i.student_numbers
        temp['create_time'] = i.create_time.strftime('%Y-%m-%d %H:%M:%S')
        res.append(temp)
    return res


# 获得数据集
def get_database_info_all():
    res = []
    data = models.database_manage2.objects.all()
    for i in data:
        temp = {}
        temp['Database_id'] = i.Database_id
        temp['Database_name'] = i.Database_name
        temp['Database_number'] = i.Database_number
        temp['Database_type'] = i.Database_type
        temp['create_time'] = i.create_time.strftime('%Y-%m-%d %H:%M:%S')
        res.append(temp)
    return res


def upload_dataset(test_name, test_path):
    flag = models.Test_dataset.objects.filter(test_name=test_name)
    if flag.count() == 0:
        # 无上传记录，新建
        models.Test_dataset.objects.create(test_name=test_name, test_path=test_path)

    else:
        # TODO:更新文件，需要删除旧文件
        report = models.Test_dataset.objects.get(test_name=test_name)
        del_url = settings.BASE_DIR + '/apps' + report.test_path
        del_url = str(del_url).replace("/", "\\")
        # print(del_url)
        os.remove(del_url)
        report.test_name = test_name
        report.test_path = test_path
        report.save()
    return 0


# ##############################################################################################################################################################


# TODO：新建班级
def group_new():
    return HttpResponse("OK")


# TODO: 删除班级
def group_delete():
    return HttpResponse("OK")


#  获取包含课程——章节的所有信息
def get_course_allinfo():
    course_list = models.Courseinfo.objects.values("course_name")
    course_list = list(course_list)
    for course in course_list:
        course_object = models.Courseinfo.objects.filter(course_name=course["course_name"])[0]
        chapter_list = models.Chapterinfo.objects.filter(course=course_object).values("chapter_id", "chapter_name")
        chapter_list = list(chapter_list)
        course["chapter"] = chapter_list
    return course_list


# 新建课程
def new_course(mcourse_name="course_demo", mteacher=""):
    # 查询最后的course_id
    last_id = models.Courseinfo.objects.all().values("course_id")
    new_id = last_id[len(last_id) - 1]['course_id'] + 1
    try:
        course = models.Courseinfo.objects.create(course_id=new_id, course_name=mcourse_name,
                                                  create_time=timezone.now(), teacher=mteacher, credit=0,
                                                  credit_hour=18)
    except Exception as e:
        print(e)
    return course.course_id


# 表格分页


# 获取该chapter_id 下对应的文件url
def get_task(chapter_id):
    # 获取chapter_id的指导书的url 之后有空再集成到utils中去
    instruct_book_url = models.Instructbook.objects.filter(chapter_id=chapter_id).values("book_url")
    instruct_book_url = list(instruct_book_url)
    instruct_book_url = instruct_book_url[0]["book_url"]
    # 获取chapter_id的实验原理的url 之后有空再集成到utils中去
    experience_url = models.Experience.objects.filter(chapter_id=chapter_id).values("experience_url")
    experience_url = list(experience_url)
    experience_url = experience_url[0]["experience_url"]
    # 获取chapter_id的视频的url 之后有空再集成到utils中去
    video_url = models.Video.objects.filter(chapter_id=chapter_id).values("video_url")
    video_url = list(video_url)
    video_url = video_url[0]["video_url"]
    # 获取chapter_id的知识点的url 之后有空再集成到utils中去
    knowledge_url = models.Knowledge.objects.filter(chapter_id=chapter_id).values("knowledge_url")
    knowledge_url = list(knowledge_url)
    knowledge_url = knowledge_url[0]["knowledge_url"]
    task_dict = {"instruct_book_url": instruct_book_url, "experience_url": experience_url, "video_url": video_url,
                 "knowledge_url": knowledge_url}
    return task_dict


# 获取该model_id 下对应的文件url
def get_model_url(model_id):
    # 获取model_id的指导书的url 之后有空再集成到utils中去
    model_url = models.model_info.objects.filter(model_id=model_id).values("model_info_url")
    model_url = list(model_url)
    model_url = model_url[0]["model_info_url"]
    task_dict = {"model_url": model_url}
    return task_dict


# 获取体系by teacher
def get_course_by_teacher(mteacher):
    course = []
    course_list = models.Courseinfo.objects.filter(teacher=mteacher)
    for i in course_list:
        data = i.to_dict()
        data["teacher"] = i.teacher.first_name
        course.append(data)
    return course


# 获取所有的体系
def get_all_course():
    course = []
    course_list = models.Courseinfo.objects.all()
    for i in course_list:
        data = i.to_dict()
        data["teacher"] = i.teacher.first_name
        course.append(data)
    return course


# 获取体系 permission=0 教师获取自己的课程 permission=1 管理员获取全部课程
def get_course(mteacher, permission=0):
    if permission == 1:
        course_list = models.Courseinfo.objects.all().values("course_id", "course_name", "create_time",
                                                             "credit", "credit_hour")
    else:
        course_list = models.Courseinfo.objects.filter(teacher=mteacher).values("course_id", "course_name",
                                                                                "create_time",
                                                                                "credit", "credit_hour")
    course_list = list(course_list)
    return course_list


# 获取章节 0 教师 1管理员
def get_chapter_by_teacher(mteacher, permission=0):
    chapter_list = []
    if permission == 1:
        # TODO：未完善
        chapter_list = models.Chapterinfo.objects.all()
        chapter_list = list(chapter_list)
    else:
        # 先获取教师的course,再获取对应的chapter信息
        course_info = get_course(mteacher)
        for course in course_info:
            chapter_info = models.Chapterinfo.objects.filter(course_id=course['course_id'])
            for chapter in chapter_info:
                temp_dict = {}
                temp_dict["chapter_id"] = chapter.chapter_id
                temp_dict["chapter_name"] = chapter.chapter_name
                temp_dict["course_name"] = course['course_name']
                temp_dict["credit"] = chapter.credit
                temp_dict["credit_hour"] = chapter.credit_hour
                # temp_dict["create_time"] = chapter.create_time
                chapter_list.append(temp_dict)
    return chapter_list


# 获取报告列表
def get_report_list(mchapter_id):
    # 初始化一个list 用来存储数据 最后并返回
    report_list = []
    # 获取report表中的所有数据
    # report_data = models.Report.objects.all()
    # 获取某一个课程章节的全部报告
    report_data = models.Report.objects.filter(chapter_id=mchapter_id)
    for i in report_data:
        # to_dict 是我重写的model_to_dict函数,model_to_dict的缺点是无法正常获取到DateTimeField类型的值
        # 使用to_dict后 外键的就变成普通的值 而不是对应的model类的对象了 所以先取一下用户的姓名和报告所属的章节
        stu_name = i.student.first_name
        chapter_name = i.chapter.chapter_name
        # 将Queryset转化成字典
        index = i.to_dict()
        # 删除一点不要的key
        del index['report_url'], index['comment']
        index['student'] = stu_name
        index['chapter'] = chapter_name
        report_list.append(index)
    return report_list


# 获取报告列表
def get_stu_report_list(report_id):
    # 初始化一个list 用来存储数据 最后并返回
    report_list = []
    # 获取report表中的所有数据
    # report_data = models.Report.objects.all()
    # 获取某一个课程章节的全部报告
    report_data = models.Report.objects.filter(report_id=report_id)
    for i in report_data:
        # to_dict 是我重写的model_to_dict函数,model_to_dict的缺点是无法正常获取到DateTimeField类型的值
        # 使用to_dict后 外键的就变成普通的值 而不是对应的model类的对象了 所以先取一下用户的姓名和报告所属的章节
        stu_name = i.student.first_name
        chapter_name = i.chapter.chapter_name
        # 将Queryset转化成字典
        index = i.to_dict()
        # 删除一点不要的key
        index['student'] = stu_name
        index['chapter'] = chapter_name
        report_list.append(index)
    return report_list


# 获取用户的个人信息
def get_user_information(username):
    user = models.UserInfo.objects.filter(username=username)[0]
    class_name = user.classroom.class_name
    user = user.to_dict()
    user["class_name"] = class_name
    user["birth"] = user["birth"].strftime('%Y-%m-%d')
    return user


# 获取下一个报告的report_id
def get_next_report(report_id):
    report = models.Report.objects.filter(report_id=report_id)[0]
    chapter = report.chapter
    report_data = models.Report.objects.filter(chapter=chapter)
    for i in range(len(report_data)):
        if (report_data[i] == report and i != len(report_data) - 1):
            return report_data[i + 1].report_id
        elif (report_data[i] == report):
            return report_data[0].report_id
        else:
            return 0


# 获取学生用户列表
def get_student_list():
    user_list = []
    user_data = models.UserInfo.objects.filter(is_superuser=0).filter(is_active=1).filter(is_staff=0)
    for i in user_data:
        index = i.to_dict()
        if index['last_login'] is None:
            index['last_login'] = "暂无数据"
        del index['password'], index['id'], index['is_superuser'], index['is_active'], index[
            'is_staff'], index['birth'], index['last_name'], index['date_joined']
        user_list.append(index)
    return user_list


# 获取教师用户列表
def get_teacher_list():
    user_list = []
    user_data = models.UserInfo.objects.filter(is_superuser=0).filter(is_active=1).filter(is_staff=1)
    for i in user_data:
        index = i.to_dict()
        if index['last_login'] is None:
            index['last_login'] = "暂无数据"
        del index['password'], index['id'], index['is_superuser'], index['is_active'], index[
            'is_staff'], index['birth'], index['last_name'], index['date_joined']
        index['classroom'] = models.Classinfo.objects.filter(class_id=index['classroom'])[0].class_name
        user_list.append(index)
    return user_list


# 获取该老师创建的课程详细内容
def get_teacher_chapter(teacher):
    chapter_list = []
    course_list = models.Courseinfo.objects.filter(teacher=teacher)
    for i in course_list:
        chapter = models.Chapterinfo.objects.filter(course=i)
        for index in chapter:
            data = index.to_dict()
            data["course_name"] = i.course_name
            chapter_list.append(data)
    return chapter_list


# 获取该老师创建的课程简单内容（课程章节对应）
def get_teacher_chapter_simple(teacher):
    chapter_list = []
    course_list = models.Courseinfo.objects.filter(teacher=teacher)
    for i in course_list:
        chapter = models.Chapterinfo.objects.filter(course=i)
        course = {}
        chapter_index_list = []
        for index in chapter:
            data = index.to_dict()
            del data["course"], data["create_time"], data["credit"], data["credit_hour"], data["knowledge_point"]
            chapter_index_list.append(data)
            course["course_name"] = i.course_name
            course["chapter_list"] = chapter_index_list
        chapter_list.append(course)
    return chapter_list


# 获取所有创建的体系
def get_all_chapter():
    chapter_list = []
    course_list = models.Courseinfo.objects.all()
    for i in course_list:
        chapter = models.Chapterinfo.objects.filter(course=i)
        for index in chapter:
            data = index.to_dict()
            data["course_name"] = i.course_name
            data["teacher"] = i.teacher.first_name
            chapter_list.append(data)
    return chapter_list


# 验证电子邮箱是否合法
def email_isvalid(email):
    regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
    if re.fullmatch(regex, email):
        return True
    else:
        return False


# 验证手机是否合法
def phone_isvalid(phone):
    regex = r"^1[356789]\d{9}$"
    if re.match(regex, phone):
        return True
    else:
        return False


# 获取教师的全部进度数据
def get_progress_data(teacher):
    res = []
    # 获取体系数据
    course_list = models.Courseinfo.objects.filter(teacher=teacher)
    # 获取章节数据
    for course in course_list:
        temp_dict1 = {}
        temp_dict1['course_id'] = course.course_id
        temp_dict1['course_name'] = course.course_name
        temp_dict1['student_num'] = 100  # 学习该体系的人数
        temp_list = []
        chapter_list = models.Chapterinfo.objects.filter(course=course)
        for chapter in chapter_list:
            temp_dict2 = {}
            temp_dict2['chapter_id'] = chapter.chapter_id
            temp_dict2['chapter_name'] = chapter.chapter_name
            # 返回人数？
            temp_dict2['progress_video'] = get_chapter_progress_data(chapter.chapter_id, 0)
            # 返回人数？
            temp_dict2['progress_report'] = get_chapter_progress_data(chapter.chapter_id, 1)

            temp_list.append(temp_dict2)
            pass
        temp_dict1['chapter_progress'] = temp_list
        res.append(temp_dict1)
    return res


# 获取单一章节的进度数据(视频+实验报告)
def get_chapter_progress_data(chapter_id, tag):
    if tag == 0:
        flag = models.Report.objects.filter(chapter=chapter_id)
        return flag.count()
    if tag == 1:
        flag = models.Learning_record.objects.filter(chapter=chapter_id)
        return flag.count()


# 根据chapter_id获取学习记录
def get_learning_record_chapter(chapter_id):
    learning_record = []
    learning_record_list = models.Learning_record.objects.filter(chapter=chapter_id)
    for index in learning_record_list:
        record = index.to_dict()
        if record['tag1']:
            record['tag1'] = 1
        else:
            record['tag1'] = 0
        if record['tag2']:
            record['tag2'] = 1
        else:
            record['tag2'] = 0
        if record['tag3']:
            record['tag3'] = 1
        else:
            record['tag3'] = 0
        if record['tag4']:
            record['tag4'] = 1
        else:
            record['tag4'] = 0
        learning_record.append(record)
    return learning_record


# 获取学习该course的人数
def get_learn_chapter_student(course_id):
    student_number = 0
    classroom_list = models.Course_Class.objects.filter(course=course_id)
    for index in classroom_list:
        student = models.UserInfo.objects.filter(classroom=index.classroom)
        student_number += len(student)
    return student_number


# 获取学习过该course各个chapter 视频的人数
def get_course_video_record(course_id):
    data_list = []
    number_list = []
    chapter_name_list = []
    chapter = models.Chapterinfo.objects.filter(course=course_id)
    # for index in chapter:
    #     chapter_id = index.chapter_id
    #     chapter_name = index.chapter_name
    #     record = {}
    #     record_data = models.Learning_record.objects.filter(chapter=index).filter(tag1=1)
    #     record["chapter_id"] = chapter_id
    #     record["chapter_name"] = chapter_name
    #     record["complete_number"] = len(record_data)
    #     data_list.append(record)
    for index in chapter:
        chapter_name = index.chapter_name
        chapter_name_list.append(chapter_name)
        record_data = models.Learning_record.objects.filter(chapter=index).filter(tag1=1)
        number_list.append(len(record_data))
    data_list.append(chapter_name_list)
    data_list.append(number_list)

    return data_list


# 获取提交过过该course各个chapter 报告的人数
def get_course_report_record(course_id):
    data_list = []
    number_list = []
    chapter_name_list = []
    chapter = models.Chapterinfo.objects.filter(course=course_id)
    # for index in chapter:
    #     chapter_id = index.chapter_id
    #     chapter_name = index.chapter_name
    #     record = {}
    #     record_data = models.Learning_record.objects.filter(chapter=index).filter(tag2=1)
    #     record["chapter_id"] = chapter_id
    #     record["chapter_name"] = chapter_name
    #     record["complete_number"] = len(record_data)
    #     data_list.append(record)
    for index in chapter:
        chapter_name = index.chapter_name
        chapter_name_list.append(chapter_name)
        record_data = models.Learning_record.objects.filter(chapter=index).filter(tag2=1)
        number_list.append(len(record_data))
    data_list.append(chapter_name_list)
    data_list.append(number_list)
    return data_list


# 获取既提交过过该course各个chapter 报告 又学习完视频的人数
def get_course_all_record(course_id):
    data_list = []
    number_list = []
    chapter_name_list = []
    chapter = models.Chapterinfo.objects.filter(course=course_id)
    # for index in chapter:
    #     chapter_id = index.chapter_id
    #     chapter_name = index.chapter_name
    #     record = {}
    #     record_data = models.Learning_record.objects.filter(chapter=index).filter(tag1=1).filter(tag2=1)
    #     record["chapter_id"] = chapter_id
    #     record["chapter_name"] = chapter_name
    #     record["complete_number"] = len(record_data)
    #     data_list.append(record)
    for index in chapter:
        chapter_name = index.chapter_name
        chapter_name_list.append(chapter_name)
        record_data = models.Learning_record.objects.filter(chapter=index).filter(tag1=1).filter(tag2=1)
        number_list.append(len(record_data))
    data_list.append(chapter_name_list)
    data_list.append(number_list)
    return data_list


# 获取所有部门
def get_all_department():
    department_list = []
    department_obeject = models.Classinfo.objects.filter(grouping=1)
    for index in department_obeject:
        index = index.to_dict()
        department_list.append(index)

    return department_list


# 学生端更新总分
def update_course_credit(stu_id, score):
    record = models.Credit_Record.objects.filter(student_id=stu_id)
    if record.count() == 0:
        models.Credit_Record.objects.create(course_credit=score, tool_credit=0, student_id=stu_id)
    else:
        record_temp = record[0]
        record_temp.course_credit = record_temp.course_credit + score
        record_temp.save()
    return 0


def update_course_record(stu, chapter_id, tag, score):
    record = models.Learning_record.objects.filter(student=stu, chapter=chapter_id)
    if record.count() == 0:
        models.Learning_record.objects.create(tag1=0, tag2=0, tag3=0, tag4=0, tag5=0, tag6=0, tag7=0, tag8=0,
                                              student=stu, chapter=chapter_id)
    record = models.Learning_record.objects.filter(student=stu, chapter=chapter_id)
    record_temp = record[0]
    if tag == 1:
        if record_temp.tag1 == 0:
            record_temp.tag1 = 1
            record_temp.save()
            print(record_temp.tag1)

            update_course_credit(stu.id, 10)
        return 0
    if tag == 2:
        if record_temp.tag2 == 0:
            record_temp.tag2 = 1
            record_temp.save()
            update_course_credit(stu.id, 10)
        return 0
    if tag == 3:
        if record_temp.tag3 == 0:
            record_temp.tag3 = 1
            record_temp.save()
            update_course_credit(stu.id, 10)
        return 0
    if tag == 4:
        if record_temp.tag4 == 0:
            record_temp.tag4 = 1
            record_temp.save()
            update_course_credit(stu.id, 10)
        return 0
    if tag == 5:
        if record_temp.tag5 != score:
            update_course_credit(stu.id, score - record[0].tag5)
            record_temp.tag5 = score
            record_temp.save()
    if tag == 6:
        if record_temp.tag6 != score:
            update_course_credit(stu.id, score - record[0].tag6)
            record_temp.tag6 = score
            record_temp.save()
        return 0
    return 0


# 学生端获取学生个人学分
def get_student_credit(stu_id):
    res = {}
    stu_credit = models.Credit_Record.objects.filter(student_id=stu_id)
    if stu_credit.count() > 0:
        res["course_credit"] = stu_credit[0].course_credit
        res["tool_credit"] = stu_credit[0].tool_credit
    else:
        models.Credit_Record.objects.create(course_credit=0, tool_credit=0, student_id=stu_id)
        res["course_credit"] = 0
        res["tool_credit"] = 0
    return res


# 教师段获取学生学分排名 总体
def get_student_rank():
    res = []
    student_list = models.Credit_Record.objects.all().order_by(F('course_credit') + F('tool_credit')).reverse()
    for item in student_list:
        temp = model_to_dict(item)
        student_obj = models.UserInfo.objects.get(id=item.student_id)
        if student_obj.is_staff == 1:
            del temp
            continue
        temp["student_id"] = student_obj.id
        temp["total"] = temp["course_credit"] + temp["tool_credit"]
        temp["username"] = student_obj.username
        temp["name"] = student_obj.first_name
        temp["class_name"] = models.Classinfo.objects.get(class_id=student_obj.classroom_id).class_name
        del temp['id']
        res.append(temp)
    return res


# 教师端获取班级排名 总体
def get_class_rank():
    # 查询班级信息
    res = []
    class_map = {}
    class_list = models.Classinfo.objects.all()
    for item in class_list:
        if item.grouping == "1":
            # 教师组移除
            continue
        class_map[item.class_id] = 0
        # temp = model_to_dict(item)
        temp = item.to_dict()
        del temp["create_time"]
        temp["credit_total"] = 0
        res.append(temp)
    record_list = models.Credit_Record.objects.all()
    for item in record_list:
        stu_info = models.UserInfo.objects.get(id=item.student_id)
        class_info = models.Classinfo.objects.get(class_id=stu_info.classroom_id)
        if not class_info.class_id in class_map:
            continue
        class_map[class_info.class_id] = class_map[class_info.class_id] + item.course_credit + item.tool_credit
    for item in res:
        item["credit_total"] = class_map[item["class_id"]]
    res.sort(key=lambda x: x["credit_total"], reverse=True)
    return res


# 获取子章节的全部学生完成情况
def get_chapter_student_progress(cha_id):
    res = []
    # 获取所有章节id, 章节记录表,
    record_list = models.Learning_record.objects.filter(chapter_id=cha_id)
    for item in record_list:
        temp = model_to_dict(item)
        student_info = models.UserInfo.objects.get(id=temp["student"])
        class_info = models.Classinfo.objects.get(class_id=student_info.classroom_id)
        temp["first_name"] = student_info.first_name
        temp["class_name"] = class_info.class_name
        res.append(temp)
    return res


# 学生章节状态重置
def reset_chapter_student_progress(record_id):
    record = models.Learning_record.objects.get(id=record_id)
    record.tag1 = 0
    record.tag2 = 0
    record.tag3 = 0
    record.tag4 = 0
    record.tag5 = 0
    record.tag6 = 0
    record.tag7 = 0
    record.tag8 = 0
    record.save()
    return 0


# 获取学生（已更新分数的）工具学分
def get_all_student_tool_score():
    res = []
    student_tool_record = models.Tool_Learing.objects.all()
    for item in student_tool_record:
        temp = model_to_dict(item)
        temp["student_name"] = models.UserInfo.objects.get(id=item.student_id).first_name
        res.append(temp)
    return res


# 获取工具完成学生的占比
def get_tool_completion():
    res = {}
    # 获取总人数
    res["student_num"] = models.UserInfo.objects.filter(is_active=1, is_staff=0).count()
    res["complete_studentnum"] = models.Credit_Record.objects.filter(
        tool_credit__gt=150).count()  # 大于__gt __gte大等 __lte
    return res


# 获取某一个班级工具完成情况比
def get_class_tool_completion(class_name):
    class_id = models.Classinfo.objects.get(class_name=class_name).class_id
    res = {}
    # 获取总人数
    res["student_num"] = models.UserInfo.objects.filter(is_active=1, is_staff=0, classroom_id=class_id).count()
    complete_studentnum = 0
    temp_list = models.Credit_Record.objects.filter(tool_credit__gt=150)
    for item in temp_list:
        stutemp = models.UserInfo.objects.get(id=item.student_id)
        if stutemp.classroom_id == class_id:
            complete_studentnum += 1
    res["complete_studentnum"] = complete_studentnum
    return res


# 获取全部题库
def get_all_question():
    question_bank = models.Question_Bank.objects.all()
    question_list = []
    for question in question_bank:
        question = question.to_dict()
        del question["answer"]
        question_list.append(question)
    return question_list


# 获取指定章节的题库
def get_chapter_question(chapter_id):
    chapter_question = models.Chapter_Question.objects.filter(chapter=chapter_id)
    question_list = []
    for question_object in chapter_question:
        question = question_object.question.to_dict()
        del question["answer"]
        question_list.append(question)
    return question_list


# 获取指定章节的题库（有答案）
def get_chapter_question_answer(chapter_id):
    chapter_question = models.Chapter_Question.objects.filter(chapter=chapter_id)
    question_list = []
    for question_object in chapter_question:
        question = question_object.question.to_dict()
        question_list.append(question)
    return question_list


# 获取指定用户创建的题库
def get_user_question(user):
    chapter_question = models.Question_Bank.objects.filter(setter=user)
    question_list = []
    for question_object in chapter_question:
        question = question_object.to_dict()
        question_list.append(question)
    return question_list


# 核对问题答案并返回分数
def get_test_score(answer_list):
    score = 0

    for index in answer_list:
        answer_id = index["id"]
        answer_type = index["question_type"]
        submit_answer = index["answer"]
        true_answer = models.Question_Bank.objects.filter(id=answer_id)[0].answer
        # ast.literal_eval：较为安全的字符串转固有类型
        true_answer = ast.literal_eval(true_answer)
        true_answer = ''.join(true_answer)
        if answer_type != '2':
            if submit_answer == true_answer:
                if answer_type == "0":
                    score += 1
                else:
                    score += 2
        else:
            # 问答题的分数，暂定2分
            score += get_QAtest_test_score(true_answer)
    return score


def get_QAtest_test_score(true_answer):
    # 教师端传分数
    return 2
