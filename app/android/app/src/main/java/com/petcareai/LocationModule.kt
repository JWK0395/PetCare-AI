package com.petcareai

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.Location
import android.location.LocationManager
import android.os.Build
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.Arguments
import java.util.Locale

/**
 * 응급 병원 검색용 위치 조회 모듈.
 *
 * 하는 일은 두 가지뿐이다.
 *   1) 마지막으로 알려진 위치(위도·경도)를 얻는다.
 *   2) Android 내장 Geocoder 로 "서울특별시 강남구" 같은 **지역명 문자열**을 만든다.
 *
 * 지역명이 필요한 이유: 병원 검색에 쓰는 Tavily 는 웹 검색 엔진이라 좌표를 이해하지
 * 못한다. "서울 강남구 24시 응급 동물병원" 같은 텍스트 질의가 있어야 결과가 나온다.
 *
 * 설계 선택
 * - **LocationManager 사용(FusedLocationProvider 아님).** Fused 는 Google Play
 *   Services 의존이라 GMS 없는 기기·에뮬레이터에서 실패한다. 응급 기능이 특정
 *   기기에서만 동작하면 안 되므로 OS 내장 API 를 쓴다.
 * - **getLastKnownLocation 만 사용.** 실시간 측위(requestLocationUpdates)는 수 초~수십 초가
 *   걸리고 실내에서는 실패한다. 응급 상황에서 기다리게 하는 것보다, 마지막 위치로 즉시
 *   지역을 잡는 편이 낫다(동/구 단위 정확도면 병원 검색에 충분하다).
 * - **권한 확인 실패·측위 실패를 예외가 아니라 결과로 돌려준다.** 앱이 "지역을 알 수
 *   없음" 경로로 자연스럽게 넘어가야 하기 때문이다(사용자가 직접 지역 입력).
 */
class LocationModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = NAME

    companion object {
        const val NAME = "PetCareLocation"
    }

    private fun hasPermission(): Boolean {
        val ctx = reactApplicationContext
        val fine = ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_FINE_LOCATION)
        val coarse = ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_COARSE_LOCATION)
        return fine == PackageManager.PERMISSION_GRANTED || coarse == PackageManager.PERMISSION_GRANTED
    }

    /**
     * 사용 가능한 provider 중 가장 최근 위치를 고른다.
     *
     * GPS 는 실내에서 오래된 값이거나 없을 수 있고, NETWORK 는 정확도는 낮지만
     * 실내에서도 잡힌다. 둘 다 확인해 **더 최근 값**을 쓴다.
     */
    private fun lastKnownLocation(): Location? {
        val manager = reactApplicationContext
            .getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            ?: return null

        var best: Location? = null
        for (provider in listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)) {
            val location = try {
                if (!manager.isProviderEnabled(provider)) continue
                manager.getLastKnownLocation(provider)
            } catch (e: SecurityException) {
                null
            } catch (e: IllegalArgumentException) {
                null
            } ?: continue

            if (best == null || location.time > best!!.time) {
                best = location
            }
        }
        return best
    }

    /**
     * 좌표 → 지역명. Tavily 질의에 넣을 수 있는 짧은 행정구역 문자열을 만든다.
     *
     * 예: "서울특별시 강남구" / "경기도 성남시 분당구"
     * 상세 주소(도로명·번지)는 넣지 않는다 — 검색어가 좁아져 결과가 오히려 줄고,
     * 사용자의 정확한 위치가 외부 검색 서비스로 나가는 것도 피하는 편이 낫다.
     */
    private fun regionName(location: Location): String? {
        if (!Geocoder.isPresent()) return null
        return try {
            val geocoder = Geocoder(reactApplicationContext, Locale.KOREA)
            @Suppress("DEPRECATION")
            val results = geocoder.getFromLocation(location.latitude, location.longitude, 1)
            val address = results?.firstOrNull() ?: return null

            // adminArea = 시/도, subAdminArea = 시/군, locality = 시, subLocality = 구/동
            val parts = listOfNotNull(
                address.adminArea,
                address.subAdminArea ?: address.locality,
                address.subLocality,
            ).distinct().filter { it.isNotBlank() }

            if (parts.isEmpty()) null else parts.take(2).joinToString(" ")
        } catch (e: Exception) {
            // Geocoder 는 네트워크·백엔드 문제로 IOException 을 자주 던진다.
            // 실패해도 좌표는 이미 얻었으므로 지역명만 비운다.
            null
        }
    }

    /**
     * 현재 지역 정보를 돌려준다. **실패해도 reject 하지 않는다.**
     *
     * 반환:
     *   { available: Boolean, latitude: Double?, longitude: Double?,
     *     regionName: String?, reason: String }
     *
     * reason 값: "ok" | "permission_denied" | "location_unavailable" | "geocoder_failed"
     * 앱은 available=false 이면 사용자에게 지역을 직접 입력받는 경로로 간다.
     */
    @ReactMethod
    fun getCurrentRegion(promise: Promise) {
        try {
            if (!hasPermission()) {
                promise.resolve(result(false, null, null, null, "permission_denied"))
                return
            }

            val location = lastKnownLocation()
            if (location == null) {
                promise.resolve(result(false, null, null, null, "location_unavailable"))
                return
            }

            val region = regionName(location)
            promise.resolve(
                result(
                    region != null,
                    location.latitude,
                    location.longitude,
                    region,
                    if (region != null) "ok" else "geocoder_failed",
                )
            )
        } catch (e: Exception) {
            // 여기까지 온 예외는 예상 밖이지만, 응급 흐름을 끊지 않는다.
            promise.resolve(result(false, null, null, null, "location_unavailable"))
        }
    }

    private fun result(
        available: Boolean,
        latitude: Double?,
        longitude: Double?,
        regionName: String?,
        reason: String,
    ): WritableMap {
        val map = Arguments.createMap()
        map.putBoolean("available", available)
        if (latitude != null) map.putDouble("latitude", latitude) else map.putNull("latitude")
        if (longitude != null) map.putDouble("longitude", longitude) else map.putNull("longitude")
        if (regionName != null) map.putString("regionName", regionName) else map.putNull("regionName")
        map.putString("reason", reason)
        return map
    }
}
