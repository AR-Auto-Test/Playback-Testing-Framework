package com.google.ar.sceneform.ux;

import android.util.Log;
import android.widget.Toast;
import com.google.ar.core.Config;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import java.util.Collections;
import java.util.Set;

/* loaded from: classes.dex */
public class ArFragment extends BaseArFragment {
    private static final String TAG = "StandardArFragment";
    public static boolean isFrontCam = false;

    @Override // com.google.ar.sceneform.ux.BaseArFragment
    public String[] getAdditionalPermissions() {
        return new String[0];
    }

    @Override // com.google.ar.sceneform.ux.BaseArFragment
    public Config getSessionConfiguration(Session session) {
        new Config(session).setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL);
        return new Config(session);
    }

    @Override // com.google.ar.sceneform.ux.BaseArFragment
    public Set<Session.Feature> getSessionFeatures() {
        return Collections.emptySet();
    }

    @Override // com.google.ar.sceneform.ux.BaseArFragment
    public void handleSessionException(UnavailableException unavailableException) {
        String str;
        if (unavailableException instanceof UnavailableArcoreNotInstalledException) {
            str = "Please install ARCore";
        } else if (unavailableException instanceof UnavailableApkTooOldException) {
            str = "Please update ARCore";
        } else if (unavailableException instanceof UnavailableSdkTooOldException) {
            str = "Please update this app";
        } else {
            str = unavailableException instanceof UnavailableDeviceNotCompatibleException ? "This device does not support AR" : "Failed to create AR session";
        }
        Log.e(TAG, "Error: " + str, unavailableException);
        Toast.makeText(requireActivity(), str, 1).show();
    }

    @Override // com.google.ar.sceneform.ux.BaseArFragment
    public boolean isArRequired() {
        return true;
    }
}