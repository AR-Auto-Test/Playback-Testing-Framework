package b.j.b;

import android.app.Notification;
import android.app.Person;
import android.app.RemoteInput;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import androidx.core.graphics.drawable.IconCompat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/* compiled from: NotificationCompatBuilder.java */
/* loaded from: classes.dex */
public class j {

    /* renamed from: a  reason: collision with root package name */
    public final Notification.Builder f2069a;

    /* renamed from: b  reason: collision with root package name */
    public final h f2070b;

    /* renamed from: c  reason: collision with root package name */
    public final List<Bundle> f2071c = new ArrayList();

    /* renamed from: d  reason: collision with root package name */
    public final Bundle f2072d = new Bundle();

    public j(h hVar) {
        List<String> list;
        Bundle bundle;
        Bundle bundle2;
        this.f2070b = hVar;
        if (Build.VERSION.SDK_INT >= 26) {
            this.f2069a = new Notification.Builder(hVar.f2060a, hVar.m);
        } else {
            this.f2069a = new Notification.Builder(hVar.f2060a);
        }
        Notification notification = hVar.o;
        this.f2069a.setWhen(notification.when).setSmallIcon(notification.icon, notification.iconLevel).setContent(notification.contentView).setTicker(notification.tickerText, null).setVibrate(notification.vibrate).setLights(notification.ledARGB, notification.ledOnMS, notification.ledOffMS).setOngoing((notification.flags & 2) != 0).setOnlyAlertOnce((notification.flags & 8) != 0).setAutoCancel((notification.flags & 16) != 0).setDefaults(notification.defaults).setContentTitle(hVar.f2064e).setContentText(hVar.f2065f).setContentInfo(null).setContentIntent(hVar.f2066g).setDeleteIntent(notification.deleteIntent).setFullScreenIntent(null, (notification.flags & 128) != 0).setLargeIcon((Bitmap) null).setNumber(0).setProgress(0, 0, false);
        this.f2069a.setSubText(null).setUsesChronometer(false).setPriority(hVar.f2067h);
        Iterator<f> it = hVar.f2061b.iterator();
        while (it.hasNext()) {
            f next = it.next();
            IconCompat a2 = next.a();
            Notification.Action.Builder builder = new Notification.Action.Builder(a2 != null ? a2.e() : null, next.j, next.k);
            m[] mVarArr = next.f2053c;
            if (mVarArr != null) {
                int length = mVarArr.length;
                RemoteInput[] remoteInputArr = new RemoteInput[length];
                if (mVarArr.length > 0) {
                    m mVar = mVarArr[0];
                    throw null;
                }
                for (int i = 0; i < length; i++) {
                    builder.addRemoteInput(remoteInputArr[i]);
                }
            }
            if (next.f2051a != null) {
                bundle2 = new Bundle(next.f2051a);
            } else {
                bundle2 = new Bundle();
            }
            bundle2.putBoolean("android.support.allowGeneratedReplies", next.f2055e);
            int i2 = Build.VERSION.SDK_INT;
            builder.setAllowGeneratedReplies(next.f2055e);
            bundle2.putInt("android.support.action.semanticAction", next.f2057g);
            if (i2 >= 28) {
                builder.setSemanticAction(next.f2057g);
            }
            if (i2 >= 29) {
                builder.setContextual(next.f2058h);
            }
            bundle2.putBoolean("android.support.action.showsUserInterface", next.f2056f);
            builder.addExtras(bundle2);
            this.f2069a.addAction(builder.build());
        }
        Bundle bundle3 = hVar.l;
        if (bundle3 != null) {
            this.f2072d.putAll(bundle3);
        }
        int i3 = Build.VERSION.SDK_INT;
        this.f2069a.setShowWhen(hVar.i);
        this.f2069a.setLocalOnly(hVar.k).setGroup(null).setGroupSummary(false).setSortKey(null);
        this.f2069a.setCategory(null).setColor(0).setVisibility(0).setPublicVersion(null).setSound(notification.sound, notification.audioAttributes);
        if (i3 < 28) {
            list = a(b(hVar.f2062c), hVar.p);
        } else {
            list = hVar.p;
        }
        if (list != null && !list.isEmpty()) {
            for (String str : list) {
                this.f2069a.addPerson(str);
            }
        }
        if (hVar.f2063d.size() > 0) {
            if (hVar.l == null) {
                hVar.l = new Bundle();
            }
            Bundle bundle4 = hVar.l.getBundle("android.car.EXTENSIONS");
            bundle4 = bundle4 == null ? new Bundle() : bundle4;
            Bundle bundle5 = new Bundle(bundle4);
            Bundle bundle6 = new Bundle();
            for (int i4 = 0; i4 < hVar.f2063d.size(); i4++) {
                String num = Integer.toString(i4);
                f fVar = hVar.f2063d.get(i4);
                Object obj = k.f2073a;
                Bundle bundle7 = new Bundle();
                IconCompat a3 = fVar.a();
                bundle7.putInt("icon", a3 != null ? a3.c() : 0);
                bundle7.putCharSequence("title", fVar.j);
                bundle7.putParcelable("actionIntent", fVar.k);
                if (fVar.f2051a != null) {
                    bundle = new Bundle(fVar.f2051a);
                } else {
                    bundle = new Bundle();
                }
                bundle.putBoolean("android.support.allowGeneratedReplies", fVar.f2055e);
                bundle7.putBundle("extras", bundle);
                bundle7.putParcelableArray("remoteInputs", k.a(fVar.f2053c));
                bundle7.putBoolean("showsUserInterface", fVar.f2056f);
                bundle7.putInt("semanticAction", fVar.f2057g);
                bundle6.putBundle(num, bundle7);
            }
            bundle4.putBundle("invisible_actions", bundle6);
            bundle5.putBundle("invisible_actions", bundle6);
            if (hVar.l == null) {
                hVar.l = new Bundle();
            }
            hVar.l.putBundle("android.car.EXTENSIONS", bundle4);
            this.f2072d.putBundle("android.car.EXTENSIONS", bundle5);
        }
        int i5 = Build.VERSION.SDK_INT;
        this.f2069a.setExtras(hVar.l).setRemoteInputHistory(null);
        if (i5 >= 26) {
            this.f2069a.setBadgeIconType(0).setSettingsText(null).setShortcutId(null).setTimeoutAfter(0L).setGroupAlertBehavior(0);
            if (!TextUtils.isEmpty(hVar.m)) {
                this.f2069a.setSound(null).setDefaults(0).setLights(0, 0, 0).setVibrate(null);
            }
        }
        if (i5 >= 28) {
            Iterator<l> it2 = hVar.f2062c.iterator();
            while (it2.hasNext()) {
                Notification.Builder builder2 = this.f2069a;
                Objects.requireNonNull(it2.next());
                builder2.addPerson(new Person.Builder().setName(null).setIcon(null).setUri(null).setKey(null).setBot(false).setImportant(false).build());
            }
        }
        if (Build.VERSION.SDK_INT >= 29) {
            this.f2069a.setAllowSystemGeneratedContextualActions(hVar.n);
            this.f2069a.setBubbleMetadata(null);
        }
    }

    public static List<String> a(List<String> list, List<String> list2) {
        if (list == null) {
            return list2;
        }
        if (list2 == null) {
            return list;
        }
        b.f.c cVar = new b.f.c(list2.size() + list.size());
        cVar.addAll(list);
        cVar.addAll(list2);
        return new ArrayList(cVar);
    }

    public static List<String> b(List<l> list) {
        if (list == null) {
            return null;
        }
        ArrayList arrayList = new ArrayList(list.size());
        for (l lVar : list) {
            Objects.requireNonNull(lVar);
            arrayList.add("");
        }
        return arrayList;
    }
}