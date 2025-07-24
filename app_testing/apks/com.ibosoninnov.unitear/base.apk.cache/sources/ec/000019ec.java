package com.google.android.play.core.assetpacks;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Parcelable;
import c.b.a.a.a;
import com.google.android.play.core.tasks.OnSuccessListener;
import com.google.android.play.core.tasks.Task;
import com.google.android.play.core.tasks.Tasks;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzaw implements zzy {
    private static final com.google.android.play.core.internal.zzag zza = new com.google.android.play.core.internal.zzag("AssetPackServiceImpl");
    private static final Intent zzb = new Intent("com.google.android.play.core.assetmoduleservice.BIND_ASSET_MODULE_SERVICE").setPackage("com.android.vending");
    private final String zzc;
    private final zzco zzd;
    private final zzeb zze;
    private com.google.android.play.core.internal.zzas zzf;
    private com.google.android.play.core.internal.zzas zzg;
    private final AtomicBoolean zzh = new AtomicBoolean();

    public zzaw(Context context, zzco zzcoVar, zzeb zzebVar) {
        this.zzc = context.getPackageName();
        this.zzd = zzcoVar;
        this.zze = zzebVar;
        if (com.google.android.play.core.internal.zzch.zzb(context)) {
            Context zza2 = com.google.android.play.core.internal.zzce.zza(context);
            com.google.android.play.core.internal.zzag zzagVar = zza;
            Intent intent = zzb;
            zzz zzzVar = zzz.zza;
            this.zzf = new com.google.android.play.core.internal.zzas(zza2, zzagVar, "AssetPackService", intent, zzzVar, null);
            this.zzg = new com.google.android.play.core.internal.zzas(com.google.android.play.core.internal.zzce.zza(context), zzagVar, "AssetPackService-keepAlive", intent, zzzVar, null);
        }
        zza.zza("AssetPackService initiated.", new Object[0]);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static Bundle zzA() {
        Bundle bundle = new Bundle();
        bundle.putInt("playcore_version_code", 11003);
        ArrayList<Integer> arrayList = new ArrayList<>();
        arrayList.add(0);
        arrayList.add(1);
        bundle.putIntegerArrayList("supported_compression_formats", arrayList);
        ArrayList<Integer> arrayList2 = new ArrayList<>();
        arrayList2.add(1);
        arrayList2.add(2);
        bundle.putIntegerArrayList("supported_patch_formats", arrayList2);
        return bundle;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static Bundle zzB(int i) {
        Bundle bundle = new Bundle();
        bundle.putInt("session_id", i);
        return bundle;
    }

    private static Task zzC() {
        zza.zzb("onError(%d)", -11);
        return Tasks.zza(new AssetPackException(-11));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zzD(int i, String str, int i2) {
        if (this.zzf != null) {
            com.google.android.play.core.tasks.zzi P = a.P(zza, "notifyModuleCompleted", new Object[0]);
            this.zzf.zzq(new zzah(this, P, i, str, P, i2), P);
            return;
        }
        throw new zzck("The Play Store app is not installed or is an unofficial version.", i);
    }

    public static /* bridge */ /* synthetic */ Bundle zzk(int i, String str, String str2, int i2) {
        Bundle zzz = zzz(i, str);
        zzz.putString("slice_id", str2);
        zzz.putInt("chunk_number", i2);
        return zzz;
    }

    public static /* bridge */ /* synthetic */ Bundle zzn(Map map) {
        Bundle zzA = zzA();
        ArrayList<? extends Parcelable> arrayList = new ArrayList<>();
        for (Map.Entry entry : map.entrySet()) {
            Bundle bundle = new Bundle();
            bundle.putString("installed_asset_module_name", (String) entry.getKey());
            bundle.putLong("installed_asset_module_version", ((Long) entry.getValue()).longValue());
            arrayList.add(bundle);
        }
        zzA.putParcelableArrayList("installed_asset_module", arrayList);
        return zzA;
    }

    public static /* bridge */ /* synthetic */ ArrayList zzv(Collection collection) {
        ArrayList arrayList = new ArrayList(collection.size());
        Iterator it = collection.iterator();
        while (it.hasNext()) {
            Bundle bundle = new Bundle();
            bundle.putString("module_name", (String) it.next());
            arrayList.add(bundle);
        }
        return arrayList;
    }

    public static /* bridge */ /* synthetic */ List zzw(zzaw zzawVar, List list) {
        ArrayList arrayList = new ArrayList();
        Iterator it = list.iterator();
        while (it.hasNext()) {
            AssetPackState next = AssetPackStates.zza((Bundle) it.next(), zzawVar.zzd, zzawVar.zze).packStates().values().iterator().next();
            if (next == null) {
                zza.zzb("onGetSessionStates: Bundle contained no pack.", new Object[0]);
            }
            if (zzbg.zza(next.status())) {
                arrayList.add(next.name());
            }
        }
        return arrayList;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static Bundle zzz(int i, String str) {
        Bundle zzB = zzB(i);
        zzB.putString("module_name", str);
        return zzB;
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final Task zza(int i, String str, String str2, int i2) {
        if (this.zzf == null) {
            return zzC();
        }
        com.google.android.play.core.tasks.zzi P = a.P(zza, "getChunkFileDescriptor(%s, %s, %d, session=%d)", new Object[]{str, str2, Integer.valueOf(i2), Integer.valueOf(i)});
        this.zzf.zzq(new zzaj(this, P, i, str, str2, i2, P), P);
        return P.zza();
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final Task zzb(List list, zzbe zzbeVar, Map map) {
        if (this.zzf == null) {
            return zzC();
        }
        com.google.android.play.core.tasks.zzi P = a.P(zza, "getPackStates(%s)", new Object[]{list});
        this.zzf.zzq(new zzaf(this, P, list, map, P, zzbeVar), P);
        return P.zza();
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final Task zzc(List list, List list2, Map map) {
        if (this.zzf == null) {
            return zzC();
        }
        com.google.android.play.core.tasks.zzi P = a.P(zza, "startDownload(%s)", new Object[]{list2});
        this.zzf.zzq(new zzac(this, P, list2, map, P, list), P);
        P.zza().addOnSuccessListener(new OnSuccessListener() { // from class: com.google.android.play.core.assetpacks.zzaa
            @Override // com.google.android.play.core.tasks.OnSuccessListener
            public final void onSuccess(Object obj) {
                AssetPackStates assetPackStates = (AssetPackStates) obj;
                zzaw.this.zzf();
            }
        });
        return P.zza();
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final Task zzd(Map map) {
        if (this.zzf == null) {
            return zzC();
        }
        com.google.android.play.core.tasks.zzi P = a.P(zza, "syncPacks", new Object[0]);
        this.zzf.zzq(new zzae(this, P, map, P), P);
        return P.zza();
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final void zze(List list) {
        if (this.zzf == null) {
            return;
        }
        com.google.android.play.core.tasks.zzi P = a.P(zza, "cancelDownloads(%s)", new Object[]{list});
        this.zzf.zzq(new zzad(this, P, list, P), P);
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final synchronized void zzf() {
        if (this.zzg == null) {
            zza.zze("Keep alive connection manager is not initialized.", new Object[0]);
            return;
        }
        com.google.android.play.core.internal.zzag zzagVar = zza;
        zzagVar.zzd("keepAlive", new Object[0]);
        if (!this.zzh.compareAndSet(false, true)) {
            zzagVar.zzd("Service is already kept alive.", new Object[0]);
            return;
        }
        com.google.android.play.core.tasks.zzi zziVar = new com.google.android.play.core.tasks.zzi();
        this.zzg.zzq(new zzak(this, zziVar, zziVar), zziVar);
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final void zzg(int i, String str, String str2, int i2) {
        if (this.zzf != null) {
            com.google.android.play.core.tasks.zzi P = a.P(zza, "notifyChunkTransferred", new Object[0]);
            this.zzf.zzq(new zzag(this, P, i, str, str2, i2, P), P);
            return;
        }
        throw new zzck("The Play Store app is not installed or is an unofficial version.", i);
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final void zzh(int i, String str) {
        zzD(i, str, 10);
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final void zzi(int i) {
        if (this.zzf != null) {
            com.google.android.play.core.tasks.zzi P = a.P(zza, "notifySessionFailed", new Object[0]);
            this.zzf.zzq(new zzai(this, P, i, P), P);
            return;
        }
        throw new zzck("The Play Store app is not installed or is an unofficial version.", i);
    }

    @Override // com.google.android.play.core.assetpacks.zzy
    public final void zzj(String str) {
        if (this.zzf == null) {
            return;
        }
        com.google.android.play.core.tasks.zzi P = a.P(zza, "removePack(%s)", new Object[]{str});
        this.zzf.zzq(new zzab(this, P, str, P), P);
    }
}