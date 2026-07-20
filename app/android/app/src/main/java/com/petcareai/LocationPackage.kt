package com.petcareai

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

/**
 * LocationModule 을 앱에 등록하는 패키지.
 *
 * autolinking 대상이 아니므로(라이브러리가 아니라 앱 내부 모듈) MainApplication 의
 * packageList 에 직접 추가한다.
 *
 * New Architecture(bridgeless) 에서도 legacy ReactPackage 는 interop 계층을 통해
 * 동작한다. TurboModule codegen 을 쓰지 않은 이유는 이 모듈이 메서드 하나뿐이라
 * spec 파일·codegen 설정을 더하는 비용이 이득보다 크기 때문이다.
 */
class LocationPackage : ReactPackage {

    override fun createNativeModules(
        reactContext: ReactApplicationContext
    ): MutableList<NativeModule> = mutableListOf(LocationModule(reactContext))

    override fun createViewManagers(
        reactContext: ReactApplicationContext
    ): MutableList<ViewManager<*, *>> = mutableListOf()
}
