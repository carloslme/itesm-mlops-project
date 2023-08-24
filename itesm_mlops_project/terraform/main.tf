
# This code is compatible with Terraform 4.25.0 and versions that are backwards compatible to 4.25.0.
# For information about validating this Terraform code, see https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

provider "google" {
  project = "itesm-mlops"
  region      = "us-central1"
}

resource "google_compute_instance" "titanic" {
  boot_disk {
    auto_delete = true
    device_name = "titanic"

    initialize_params {
      image = "projects/debian-cloud/global/images/debian-11-bullseye-v20230814"
      size  = 10
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }


  # Install Flask
  metadata_startup_script = <<-EOT
    #!/bin/bash
    export CLOUDSDK_CORE_DISABLE_PROMPTS=1
    # Execute the script commands and redirect the output to the log file
    echo "$(date +"%Y-%m-%d %H:%M:%S") | Initializing setup" >> "/tmp/file.log" 2>&1
    sudo apt-get update >> "/tmp/file.log" 2>&1
    sudo apt-get install -yq build-essential python3-pip rsync >> "/tmp/file.log" 2>&1
    sudo apt-get install docker.io -yq >> "/tmp/file.log" 2>&1
    sudo usermod -aG docker $USER >> "/tmp/file.log" 2>&1
    
    sudo chown $USER /var/run/docker.sock

    sudo curl -L "https://github.com/docker/compose/releases/download/1.26.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo git clone https://github.com/carloslme/itesm-mlops-project.git
    sudo docker network create AIservice
    
    echo "$(date +"%Y-%m-%d %H:%M:%S") | Setup finished" >> "/tmp/file.log" 2>&1
    touch /tmp/startup-script-complete

    sudo docker-compose -f itesm-mlops-project/itesm_mlops_project/docker-compose.yml up --build
        
  EOT

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  labels = {
    goog-ec-src = "vm_add-tf"
  }

  machine_type = "f1-micro"
  name         = "titanic"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    subnetwork = "projects/itesm-mlops/regions/us-central1/subnetworks/default"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = "558176708918-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  tags = ["http-server", "https-server", "titanic-firewall", "default-allow-ssh"]
  zone = "us-central1-a"

}

output "public_ip" {
  value = google_compute_instance.titanic.network_interface[0].access_config[0].nat_ip
}

