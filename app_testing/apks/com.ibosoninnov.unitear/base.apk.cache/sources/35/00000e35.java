package com.google.android.gms.common.api.internal;

import android.app.Dialog;
import android.app.PendingIntent;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GoogleApiAvailability;
import com.google.android.gms.common.api.GoogleApiActivity;
import com.google.android.gms.common.internal.Preconditions;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zal implements Runnable {
    public final /* synthetic */ zak zaa;
    private final zam zab;

    public zal(zak zakVar, zam zamVar) {
        this.zaa = zakVar;
        this.zab = zamVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        if (this.zaa.zaa) {
            ConnectionResult zab = this.zab.zab();
            if (zab.hasResolution()) {
                zak zakVar = this.zaa;
                zakVar.mLifecycleFragment.startActivityForResult(GoogleApiActivity.zaa(zakVar.getActivity(), (PendingIntent) Preconditions.checkNotNull(zab.getResolution()), this.zab.zaa(), false), 1);
                return;
            }
            zak zakVar2 = this.zaa;
            if (zakVar2.zac.getErrorResolutionIntent(zakVar2.getActivity(), zab.getErrorCode(), null) != null) {
                zak zakVar3 = this.zaa;
                zakVar3.zac.zaa(zakVar3.getActivity(), this.zaa.mLifecycleFragment, zab.getErrorCode(), 2, this.zaa);
            } else if (zab.getErrorCode() == 18) {
                Dialog zaa = GoogleApiAvailability.zaa(this.zaa.getActivity(), this.zaa);
                zak zakVar4 = this.zaa;
                zakVar4.zac.zaa(zakVar4.getActivity().getApplicationContext(), new zan(this, zaa));
            } else {
                this.zaa.zaa(zab, this.zab.zaa());
            }
        }
    }
}