
# 졸업 프로젝트 : 단안 깊이 추정(Monocular Depth Estimation) 모델의 객체 깊이 추정 성능 개선 시도

건국대학교 전기전자공학부 학부 졸업 프로젝트로서 단안 깊이 추정(Monocular Depth Estimation) 모델의 전반적인 깊이 맵의 추정 정확도, 동적 객체에 대한 깊이 추정 성능을 개선하기 위한 아이디어를 연구하였습니다.
<br/><br/>

## 목차
- [프로젝트 소개](#프로젝트-소개)
- [기술 스택](#기술-스택)
- [개발 과정](#개발-과정)
- [결과 및 성과](#결과-및-성과)
- [참고 문헌](#참고-문헌)
<br/><br/>

## 프로젝트 소개
- **주제:** 단안 깊이 추정(Monocular Depth Estimation) 모델의 성능 개선 시도
- **목표:** 전반적인 깊이 맵의 추정 정확도 개선, 동적 객체에 대한 깊이 추정 개선
<br/><br/>

## 기술 스택
- Python
- Pytorch
<br/><br/>

## 개발 과정
- **기반 모델:** Monodepth2 (Digging into Self-Supervised Monocular Depth Estimation, ICCV 2019)
- **개발 목표:** 전반적인 깊이 맵의 추정 정확도 개선, 동적 객체에 대한 깊이 추정 개선
- **아이디어:** Channel Attention Module괴 Spatial Attention Module을 Network에 추가 
<br/><br/>

## 결과 및 성과
|                        | abs_rel | sq_rel |  rmse  | rmse_log | δ < 1.25 | δ < 1.25^2 | δ < 1.25^3 |
|------------------------|---------|--------|--------|----------|----------|------------|------------|
|       Monodepth2       | 0.127   | 0.974  | 5.046  | 0.202    | 0.858    | 0.956      | 0.980      |
| Monodepth2 + Attention | 0.123   | 1.012  | 5.011  | 0.201    | 0.860    | 0.955      | 0.981      |

<br/><br/>

## 참고 문헌
- Digging into Self-Supervised Monocular Depth Estimation, ICCV 2019
- CBAM : Convolutional Block Attention Moculde, ECCV 2018
