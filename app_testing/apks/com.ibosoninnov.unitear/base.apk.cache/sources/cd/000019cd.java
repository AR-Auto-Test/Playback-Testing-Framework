package com.google.android.play.core.assetpacks;

import android.os.Bundle;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public abstract class AssetPackStates {
    public static AssetPackStates zza(Bundle bundle, zzco zzcoVar, zzeb zzebVar) {
        return zzd(bundle, zzcoVar, zzebVar, new ArrayList(), zzbf.zza);
    }

    public static AssetPackStates zzb(Bundle bundle, zzco zzcoVar, zzeb zzebVar, zzbe zzbeVar) {
        return zzd(bundle, zzcoVar, zzebVar, new ArrayList(), zzbeVar);
    }

    public static AssetPackStates zzc(Bundle bundle, zzco zzcoVar, zzeb zzebVar, List list) {
        return zzd(bundle, zzcoVar, zzebVar, list, zzbf.zza);
    }

    private static AssetPackStates zzd(Bundle bundle, zzco zzcoVar, zzeb zzebVar, List list, zzbe zzbeVar) {
        ArrayList<String> stringArrayList = bundle.getStringArrayList("pack_names");
        HashMap hashMap = new HashMap();
        int size = stringArrayList.size();
        for (int i = 0; i < size; i++) {
            String str = stringArrayList.get(i);
            hashMap.put(str, AssetPackState.zzc(bundle, str, zzcoVar, zzebVar, zzbeVar));
        }
        Iterator it = list.iterator();
        while (it.hasNext()) {
            String str2 = (String) it.next();
            hashMap.put(str2, AssetPackState.zzb(str2, 4, 0, 0L, 0L, ShadowDrawableWrapper.COS_45, 1, "", ""));
        }
        return new zzbo(bundle.getLong("total_bytes_to_download"), hashMap);
    }

    public abstract Map<String, AssetPackState> packStates();

    public abstract long totalBytes();
}